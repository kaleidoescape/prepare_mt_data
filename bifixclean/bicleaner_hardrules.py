#!/usr/bin/env python

import argparse
import io
import logging
import os
import pycld2
import regex
import sys
import traceback
import yaml
import fasttext
import typing

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import TemporaryFile, NamedTemporaryFile, gettempdir
from timeit import default_timer

regex_blank = regex.compile("[ \u00A0]")
regex_digit = regex.compile("[[:digit:]]")
regex_punct = regex.compile("[[:punct:]]")
regex_alpha = regex.compile("[[:alpha:]]")
regex_url = regex.compile('((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\((:?[^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
#regex_breadcrumbs = regex.compile("([ ][-/»][ ]|[|<>→←]|[ ][:][:][ ])")
regex_breadcrumbs1 = regex.compile("([ ][-/][ ]|[<>*]|[ ][:][ ])")
regex_breadcrumbs2 = regex.compile("([ ][»][ ]|[|→←•·¬])")
regex_unicode_noise = regex.compile("[\x80-\xFF]{3,}")
regex_spaces_noise = regex.compile("([ ].){4,}[ ]")
regex_paren = regex.compile("[][(){}]")
regex_unwanted = regex.compile("[+*]")
regex_inconditional = regex.compile("=\"")
regex_escaped_unicode = regex.compile("[\\\\]u[0-9a-fA-F]{3,}")
#regex_glued_words = regex.compile("\b[[:alpha:]]*[[:lower:]][[:upper:]][[:alpha:]]*)
regex_glued_words = regex.compile("([[:alpha:]]*[[:upper:]]{1}[[:lower:]]+){3}")
safe_noise_detection_langs = {"en", "es", "fr", "pl", "de", "it", "pt", "nl", "cs", "ro", "fi", "lv", "et", "bg", "hr", "da", "hu", "ga", "eu", "gl", "sl", "sv", "mt", "sk"}

safe_noise_detection_langs = {"en", "es", "fr", "pl", "de", "it", "pt", "nl", "cs", "ro", "fi", "lv", "et", "bg", "hr", "da", "hu", "ga", "eu", "gl", "sl", "sv", "mt", "sk", "is", "lt", "nb", "nn", "no"}
similar_pairs = [{"es","ca"}, {"es","gl"}, {"pt","gl"}, {"no","nn"}, {"no", "da"}]

logging_level = 0

# Logging config
def logging_setup(args = None):
    logger = logging.getLogger()
    logger.handlers = [] # Removing default handler to avoid duplication of log messages
    logger.setLevel(logging.ERROR)
    
    h = logging.StreamHandler(sys.stderr)
    if args != None:
       h = logging.StreamHandler(args.logfile)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)

    if args != None:
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)
    logging_level = logging.getLogger().level

# Check if the argument of a program (argparse) is positive or zero
def check_positive_between_zero_and_one(value):
    ivalue = float(value)
    if ivalue < 0 or ivalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid float value between 0 and 1" % value)
    return ivalue

# Check if the argument of a program (argparse) is positive or zero
def check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_if_folder(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("%s is not a directory" % path)
    return path


def shuffle_file(input: typing.TextIO, output: typing.TextIO):
    offsets=[]
    with TemporaryFile("w+") as temp:
        count = 0
        for line in input:
            offsets.append(count)
            count += len(bytearray(line, "UTF-8"))
            temp.write(line)
        temp.flush()
        
        random.shuffle(offsets)
        
        for offset in offsets:
            temp.seek(offset)
            output.write(temp.readline())

def initialization():
    global logging_level
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('input',  nargs='?', type=argparse.FileType('rt', errors="replace"), default=io.TextIOWrapper(sys.stdin.buffer, errors="replace"),  help="Tab-separated bilingual tagged file")
    parser.add_argument('output', nargs='?', type=argparse.FileType('wt'), default=sys.stdout, help="Output of the classification")
    parser.add_argument('--annotated_output',default=False, action='store_true', help="Adds an extra column with each sentence's evaluation (\"keep\" if the sentence is good, otherwise the reason for rejecting")
    
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")

    groupO.add_argument('--disable_lang_ident', default=False, action='store_true', help="Don't apply rules that use language detecting")
    groupO.add_argument('--disable_minimal_length', default=False, action='store_true', help="Don't apply minimal length rule")

    groupO.add_argument("-s", "--source_lang", type=str, default=None,  help="Source language (SL) of the input")
    groupO.add_argument("-t", "--target_lang", type=str, default=None,  help="Target language (TL) of the input")

    groupO.add_argument("--scol", default=1, type=check_positive, help ="Source sentence column (starting in 1)")
    groupO.add_argument("--tcol", default=2, type=check_positive, help ="Target sentence column (starting in 1)")  
    
    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()
    logging_setup(args)
    
    logging_level = logging.getLogger().level
    
    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    return args
    
def c_identical(left, right, left_lang, right_lang):
    if left_lang =="nb":
        left_lang="no"
    if right_lang=="nb":
        right_lang="no"
#    if ({left_lang, right_lang} in similar_pairs):        
#        return True
    return left.casefold() != right.casefold()
    
def c_identical_wo_digits(left, right, left_lang, right_lang):
    left = regex_digit.sub("", left)
    right = regex_digit.sub("", right)
    return c_identical(left, right, left_lang, right_lang)

def c_identical_wo_punct(left, right, left_lang, right_lang):
    left = regex_punct.sub("", left)
    right = regex_punct.sub("", right)
    return c_identical(left, right, left_lang, right_lang)
        
def c_minimal_length(sentence):
    """ Counts number of whitespace, requires >= 2 (3 words) """
    return len(regex_blank.findall(sentence)) >= 2
        
def c_length(left, right):
    return 0.5 <= float(len(left))/float(len(right)) <= 2.0

def c_length_bytes(left, right):
    return 0.5 <= float(len(left.encode("utf8")))/float(len(right.encode("utf8"))) <= 2.0

def c_different_language(left, right, left_lang, right_lang):
    if left_lang =="nb":
        left_lang="no"

    if right_lang=="nb":
        right_lang="no"
        

    l_reliable = False
    l_bytes = 0
    l_details = ()
 
    try:
        l_reliable, l_bytes, l_details = pycld2.detect(left)
    except:
        return False # encoding error -> noise

    r_reliable = False
    r_bytes = 0
    r_details = ()

    try:
        r_reliable, r_bytes, r_details = pycld2.detect(right)
    except:
        return False # encoding error -> noise
        
    if l_reliable and r_reliable and l_details[0][1] != r_details[0][1]:    
        return True
    elif not l_reliable or not r_reliable:
        return True
    else:
        #both langs are reliable at this point, and the identified language is the same for left and right
        identified = l_details[0][1]
        if (identified in [left_lang, right_lang]  and {left_lang, right_lang} in similar_pairs):
            return True
        else:    
            return False
        
def c_reliable_long_language(sentence, language):
    if language=="nb":
        language = "no"
        
    reliable = False
    bytes = 0
    details = ()
    
    try:
        reliable, bytes, details = pycld2.detect(sentence)
    except:
        return True # encoding error -> noise
    
    if len(sentence) > 30 and reliable and details[0][1] != language:
        if {language, details[0][1]} in similar_pairs:
            return True
        else:
            return False
    else:
        return True
        
def c_alpha(sentence):
    return len(regex_alpha.findall(sentence)) > 0
    
def c_majority_alpha(sentence):
    return float(len(regex_alpha.findall(sentence))) / float(len(sentence)) >= 0.5

def c_no_urls(sentence):
    return sum([len("".join(i)) for i in regex_url.findall(sentence)]) < 15

#def c_no_breadcrumbs(sentence):
#    return len(regex_breadcrumbs.findall(sentence)) < 3


def c_no_breadcrumbs1(sentence):
    return len(regex_breadcrumbs1.findall(sentence)) < 3  

def c_no_breadcrumbs2(sentence):
    return len(regex_breadcrumbs2.findall(sentence)) < 2  

def c_no_noise(sentence):
    return len(regex_unicode_noise.findall(sentence)) == 0
    
def c_no_space_noise(sentence):
    return len(regex_spaces_noise.findall(sentence)) == 0
    
def c_no_paren(sentence):
    return len(regex_paren.findall(sentence)) < 10

def c_unwanted(sentence):
    return len(regex_unwanted.findall(sentence)) < 5

def c_inconditional(sentence):
    return len(regex_inconditional.findall(sentence)) < 1

def c_no_literals(literals, sentence):
    return not any(l in sentence for l in literals)

def c_no_escaped_unicode(sentence):
    return len(regex_escaped_unicode.findall(sentence)) == 0

def c_no_glued_words(sentence):
    return regex_glued_words.search(sentence) == None

def wrong_tu(left, right, args):
    if len(left) >= 1024:
        return "len(left) >= 1024"
    if len(right) >= 1024:
        return "len(right) >= 1024"
    elif not c_no_literals(["Re:"], left):
        return "c_no_literals(['Re:'], left)"
    elif not c_no_literals(["Re:"], right):
        return "c_no_literals(['Re:'], right)"            
    elif not args.disable_minimal_length and not (c_minimal_length(left) or c_minimal_length(right)):
        return "c_minimal_length(left) and c_minimal_length(right)"
    elif not (c_length(left, right) or c_length_bytes(left, right)): 
        return "c_length or c_length_bytes"
    elif not c_identical(left, right, args.source_lang, args.target_lang):
        return "c_identical"
    elif not c_identical_wo_digits(left, right, args.source_lang, args.target_lang):
        return "c_identical_wo_digits"    
    elif not c_identical_wo_punct(left, right, args.source_lang, args.target_lang):
        return "c_identical_wo_punct"    
    elif (not args.disable_lang_ident and not  c_different_language(left, right, args.source_lang, args.target_lang)):
        return "c_different_language"
    elif not c_majority_alpha(left):
        return "c_majority_alpha(left)"
    elif not c_majority_alpha(right):
        return "c_majority_alpha(right)"
    elif not c_no_urls(left):
        return "c_no_urls(left)"
    elif not c_no_urls(right):
        return "c_no_urls(right)"
    #elif not c_no_breadcrumbs(left):    
    #    return "c_no_breadcrumbs(left)"
    #elif not c_no_breadcrumbs(right):
    #    return "c_no_breadcrumbs(right)"
    elif not c_no_breadcrumbs1(left):
        return "c_no_breadcrumbs1(left)"
    elif not c_no_breadcrumbs1(right):
        return "c_no_breadcrumbs1(right)"
    elif not c_no_breadcrumbs2(left):
        return "c_no_breadcrumbs2(left)"
    elif not c_no_breadcrumbs2(right):
        return "c_no_breadcrumbs2(right)"       
    elif not c_no_glued_words(left):
        return "c_no_glued_words(left)"
    elif not c_no_glued_words(right):
        return "c_no_glued_words(right)"    
    elif args.source_lang in safe_noise_detection_langs and not c_no_noise(left):
        return "args.source_lang in safe_noise_detection_langs and not c_no_noise(left)" 
    elif args.target_lang in safe_noise_detection_langs and not c_no_noise(right):
        return "args.target_lang in safe_noise_detection_langs and not c_no_noise(right)"
    elif not c_no_space_noise(left):
        return "c_no_space_noise(left)"
    elif not c_no_space_noise(right):
        return "c_no_space_noise(right)"
    elif not c_no_paren(left):
        return "c_no_paren(left)"
    elif not c_no_paren(right):
        return "c_no_paren(right)"
    elif not c_unwanted(left):
        return "c_unwanted(left)"
    elif not c_unwanted(right):
        return "c_unwanted(right)"
    elif not c_inconditional(left):
        return "c_inconditional(left)"
    elif not c_inconditional(right):
        return "c_inconditional(right)"
    elif not c_no_escaped_unicode(left):
        return "c_no_escaped_unicode(left)"
    elif not c_no_escaped_unicode(right):
        return "c_no_escaped_unicode(right)"
    elif not c_no_literals(["{{", "%s", "}}"], left):
        return 'c_no_literals(["{{", "%s", "}}"], left)'
    elif not c_no_literals(["{{", "%s", "}}"], right):
        return 'c_no_literals(["{{", "%s", "}}"], right)'
    elif left.istitle() and right.istitle():
        return 'left.istitle() and right.istitle()'
    elif (not args.disable_lang_ident and not  c_reliable_long_language(left, args.source_lang)):
        return "c_reliable_long_language(left, sourcelang)"
    elif (not args.disable_lang_ident and  not c_reliable_long_language(right, args.target_lang)):
        return "c_reliable_long_language(right, targetlang)"
    return False
    
def reduce_process(output_queue, args):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    args.output.write(i)
                filein.close()
            os.unlink(filein_name)

        job = output_queue.get()
        if job:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug("Still elements in heap")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heapq.heappop(h)
        last_block += 1

        with open(filein_name, 'r') as filein:
            for i in filein:
                args.output.write(i)
            filein.close()

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Hard rules applied. Output available in {}".format(args.output.name))
    args.output.close()
    
def worker_process(i, jobs_queue, output_queue, args):
    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))

                for i in filein:	
                    parts = i.strip().split("\t")
                    left = ""
                    right= ""
                    
                    if len(parts) >=  args.scol and len(parts) >= args.tcol:
                        left = parts[args.scol-1]
                        right = parts[args.tcol-1]
                    else:
                        logging.error("WARNING: scol ({}) or tcol ({}) indexes above column number ({})".format(args.scol, args.tcol, len(parts)))        
                        continue
                    wrong_tu_results = wrong_tu(left,right, args)
                    if wrong_tu_results != False:
                        fileout.write("\t".join(parts)+"\t0")
                        if args.annotated_output:                            
                            fileout.write("\t{}\n".format(wrong_tu_results))
                        else:
                            fileout.write("\n")
                    else:
                        fileout.write("\t".join(parts)+"\t1")
                        if args.annotated_output:
                            fileout.write("\tkeep\n")
                        else:
                            fileout.write("\n")    

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()

            if ojob:                    
                output_queue.put(ojob)

            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker")
            break

def mapping_process(args, jobs_queue):
    logging.info("Start mapping")
    nblock = 0
    nline = 0
    mytemp = None
    for line in args.input:
        if (nline % args.block_size) == 0:
            logging.debug("Creating block {}".format(nblock))
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)
        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()        
        jobs_queue.put(job)

    return nline
        
def perform_hardrules_filtering(args):
    time_start = default_timer()
    logging.info("Starting process")
    logging.info("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count

    # Start reducer
    reduce = Process(target = reduce_process,
                     args   = (output_queue, args))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []
    for i in range(worker_count):
        filter = Process(target = worker_process,
                         args   = (i, jobs_queue, output_queue, args))
        filter.daemon = True # dies with the parent process

        filter.start()
        workers.append(filter)

    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)
    args.input.close()

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End mapping")

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()
    
    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))

def main(args):
    logging.info("Executing main program...")
    perform_hardrules_filtering(args)
    logging.info("Program finished")

if __name__ == '__main__':
    try: 
        logging_setup()
        args = initialization()
        main(args)
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)