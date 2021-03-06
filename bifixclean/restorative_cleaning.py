#!/usr/bin/env python
import ftfy
import os
import regex
import re
import html

global_chars_lang = {}


def getCharsReplacements(lang):
    # languages that use cyrillic alphabet
    # Check https://www.tug.org/TUGboat/tb17-2/tb51pisk.pdf and/or
    # https://www.quora.com/Which-languages-are-written-in-Cyrillic-script
    # for a comprehensive list
    # Some of the langs does accept both cyrillic and latin, or is migrating to latin
    cyrillic_langs = [
        "ab",  # abkhazian
        "av",  # avar/avaric
        "az",  # azerbaijani
        "ba",  # bashkir
        "be",  # belarusian
        "bg",  # bulgarian
        "bs",  # bosnian
        "ce",  # chechen
        "cv",  # chuvash
        "kk",  # kazakh
        "ku",  # kurdish
        "kv",  # komi
        "ky",  # kirghiz/kyrgyz
        "mk",  # macedonian
        "mn",  # mongolian
        "os",  # ossetic/ossetian
        "ru",  # russian
        "sr",  # serbian
        "tg",  # tajik/tadzhik
        "tk",  # turkmen
        "tt",  # tatar
        "ug",  # uighur
        "uk",  # ukranian
        "uz"  # uzbek
    ]

    # https://en.wikipedia.org/wiki/Caron
    # http://diacritics.typo.cz/index.php?id=5
    langs_with_carons = [
        "cs",  # czech
        "et",  # estonian
        "fi",  # finnish
        "hr",  # croatian
        "ln",  # lingala
        "lt",  # lithuanian
        "lv",  # latvian
        "sk",  # slovak
        "sl",  # slovenian
        "sr",  # serbian
        "yo"  # yoruba
    ]

    # Annoying characters, common for all languages
    chars = {
        '\u2028': ' ',  # line separators (\n)
        '&#10;': "",  # \n
        '\n': "",
        '&#xa': "",
        '&#xA': "",

        '\u000D': "",  # carriage returns (\r)
        '&#13;': " ",
        '&#xd;': " ",
        '&#xD;': " ",

        # unicode ligatures
        '\uFB00': 'ff',
        '\uFB01': 'fi',
        '\uFB02': 'fl',
        '\uFB03': 'ffi',
        '\uFB04': 'ffl',
        '\uFB06': 'st',

        '&nbsp;': " ",
        '&lt;': "<",
        '&gt;': ">",
        '&amp;': "&",
        '&quot;': '"',
        '&apos;': "'",
        '&iexcl;': '??',
        '&cent;': '??',
        '&pound;': '??',

        '&Agrave;': '??',
        '&Aacute;': '??',
        '&Acirc;': '??',
        '&Atilde;': '??',
        '&Auml;': '??',
        '&Aring;': '??',
        '&Aelig;': '??',

        '&agrave;': '??',
        '&aacute;': '??',
        '&acirc;': '??',
        '&atilde;': '??',
        '&auml;': '??',
        '&aring;': '??',
        '&aelig;': '??',

        '&Ccedil;': '??',
        '&ccedil;': '??',

        '&Egrave;': '??',
        '&Eacute;': '??',
        '&Ecirc;': '??',
        '&Euml;': '??',

        '&egrave;': '??',
        '&eacute;': '??',
        '&ecirc;': '??',
        '&euml;': '??',
        '&Igrave;': '??',
        '&Iacute;': '??',
        '&Icirc;': '??',
        '&Iuml;': '??',

        '&igrave;': '??',
        '&iacute;': '??',
        '&icirc;': '??',
        '&iuml;': '??',

        '&Ntilde;': '??',
        '&ntilde;': '??',

        '&Ograve;': '??',
        '&Oacute;': '??',
        '&Ocirc;': '??',
        '&Otilde;': '??',
        '&Ouml;': '??',

        '&ograve;': '??',
        '&oacute;': '??',
        '&ocirc;': '??',
        '&otilde;': '??',
        '&ouml;': '??',

        '&times;': '??',  # ??
        '&Oslash;': '',  # ??
        '&oslash;': '',  # ??

        '&Ugrave;': '??',
        '&Uacute;': '??',
        '&Ucirc;': '??',
        '&Uuml;': '??',

        '&ugrave;': '??',
        '&uacute;': '??',
        '&ucirc;': '??',
        '&uuml;': '??',

        '&Yacute;': '??',
        '&yacute;': '??',
        '&yuml;': '??',  # ??
        '&Yuml;': '??',  # capital Y with diaeres ??

        '&thorn;': '??',  # ??
        '&szlig;': '??',  # ??

        '&divide;': '??',  # ??
        '&euro;': '???',

        '\u02C1': "??",  # ?? -> ?
        '\u02C2': "<",  # ?? -> <
        '\u02C3': ">",  # ?? -> >

        # https://www.utf8-chartable.de/unicode-utf8-table.pl?number=1024&names=2&utf8=char

        '????': '??',
        '????': '??',

        '?????': '??',
        '??<80>': '??',
        '??<81>': '??',
        '??<82>': '??',
        '??<83>': '??',
        '??<84>': '??',
        '??<85>': '??',
        '?????': '??',
        '????': '??',
        '?????': '??',
        '?????': '??',
        '??<86>': '??',
        '?????': '??',
        '??<80>': '??',
        '??<82>': '??',

        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',
        '??<81>': '??',
        '??<83>': '??',

        '??<87>': '??',
        '?????': '??',
        '????': '??',

        '??<88>': '??',
        '??<89>': '??',
        '??<8A>': '??',
        '??<8B>': '??',
        '????': '??',
        '?????': '??',
        '????': '??',
        '?????': '??',

        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',

        '??<8C>': '??',
        '??<8E>': '??',
        '????': '??',
        '??<8D>': '??',
        '????': '??',
        '??<8F>': '??',
        '????': '??',
        '??<AD>': '??',
        '????': '??',
        '????': '??',

        '????': '??',

        '?????': '??',
        '??<92>': '??',
        '??<93>': '??',
        '??<94>': '??',
        '??<95>': '??',
        '??<96>': '??',
        '?????': '??',
        '?????': '??',
        '?????': '??',
        '?????': '??',
        '????': '??',
        '??<98>': '??',
        '?????': '??',

        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',
        '?????': '??',
        '????': '??',

        '?????': '??',
        '??<91>': '??',
        '????': '??',

        '????': '??',

        '????': '??',

        '??<99>': '??',
        '??<9A>': '??',
        '??<9B>': '??',
        '??<9C>': '??',
        '?????': '??',
        '????': '??',
        '?????': '??',
        '????': '??',

        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',

        '??<9D>': '??',
        '????': '??',

        '????': '??',
        '????': '??',
        '????': '??',
        '????': '??',

        '????': '??',
        '??<9E>': '??',
        '????': '??',

        '??<90>': '??',
        '????': '??',
        '??<9F>': '??',
        '????': '??',
        '????': '??'

    }

    if lang.lower() not in langs_with_carons:
        chars['\u0165'] = "t'"  # latin small letter t with caron ??
        chars['\u0192'] = "f"  # latin small letter f with hook ??

    if lang.lower() not in cyrillic_langs:
        # Cyrilic charcaters replaced to latin characters
        chars['??'] = '??'
        chars['??'] = '??'
        chars['??'] = 'S'
        chars['??'] = '??'
        chars['??'] = 'I'
        chars['??'] = 'J'
        chars['??'] = 'A'
        chars['??'] = 'B'
        chars['??'] = 'E'
        chars['??'] = 'K'
        chars['??'] = 'M'
        chars['??'] = 'H'
        chars['??'] = 'O'
        chars['??'] = 'P'
        chars['??'] = 'C'
        chars['??'] = 'T'
        chars['??'] = 'y'
        chars['??'] = 'X'
        chars['??'] = 'b'
        chars['??'] = '??'
        chars['??'] = '??'
        chars['??'] = 'a'
        chars['??'] = 'B'
        chars['??'] = 'r'
        chars['??'] = 'e'
        chars['??'] = 'k'
        chars['??'] = 'M'
        chars['??'] = 'H'
        chars['??'] = 'o'
        chars['??'] = 'p'
        chars['??'] = 'c'
        chars['??'] = 'T'
        chars['??'] = 'y'
        chars['??'] = 'x'
        chars['??'] = 'b'
        chars['??'] = 's'
        chars['??'] = 'i'
        chars['??'] = '??'
        chars['??'] = 'j'
        chars['??'] = 'Y'
        chars['??'] = 'Y'
        chars['??'] = 'h'
        chars['??'] = 'h'
        chars['??'] = 'I'
        chars['??'] = 'I'
        chars['??'] = '??'
        chars['??'] = '??'
        chars['??'] = 'd'
        chars['??'] = 'd'
        chars['??'] = 'Q'
        chars['??'] = 'q'
        chars['??'] = 'W'
        chars['??'] = 'w'
        chars['???'] = 'l'
        chars['???'] = 'l'
        chars['???'] = 'S'
        chars['???'] = 'S'
        chars['\u0443'] = 'y'  #

    if lang.lower() == "de":
        # Remove and/or replace certain keys from 'chars' in German
        chars['&bdquo;'] = '???'
        chars['\u201E'] = '???'
        chars['\u00D8'] = '??'  # latin capital letter o with stroke ??
        chars['\u00F8'] = '??'  # latin small letter o with stroke ??
        chars['&Oslash;'] = '??'  # ??
        chars['&oslash;'] = '??'
    else:
        chars['??'] = "'"
        chars['??'] = '"'
        chars['??'] = '-'
        chars['??'] = '|'
        chars['??'] = ':'

    if lang.lower() != "el":
        # Greek Letters
        chars['&Alpha;'] = 'A'  # Alpha   ?? -> Changed to latin A
        chars['??'] = 'A'  # Alpha  ?? -> Changed to latin A
        chars['&Beta;'] = 'B'  # Beta     ?? -> Changed to latin B
        chars['??'] = 'B'  # Beta   ?? -> Changed to latin B
        chars['&Gamma;'] = '??'  # Gamma   ??
        chars['&Delta;'] = '??'  # Delta   ??
        chars['&Epsilon;'] = 'E'  # Epsilon       ?? -> Changed to latin E
        chars['??'] = 'E'  # Epsilon        ?? -> Changed to latin E
        chars['&Zeta;'] = 'Z'  # Zeta     ?? -> Changed to latin Z
        chars['??'] = 'Z'  # Zeta   ?? -> Changed to latin Z
        chars['&Eta;'] = 'H'  # Eta       ?? -> Changed to latin H
        chars['??'] = 'H'  # Eta    ?? -> Changed to latin H
        chars['&Theta;'] = '??'  # Theta   ??
        chars['&Iota;'] = 'I'  # Iota     ?? -> Chaged to latin I
        chars['??'] = 'I'  # Iota   ?? -> Chaged to latin I
        chars['&Kappa;'] = 'K'  # Kappa   ?? -> Changed to latin K
        chars['??'] = 'K'  # Kappa  ?? -> Changed to latin K
        chars['&Lambda;'] = '??'  # Lambda ??
        chars['&Mu;'] = 'M'  # Mu ?? -> Changed to latin M
        chars['??'] = 'M'  # Mu     ?? -> Changed to latin M
        chars['&Nu;'] = 'N'  # Nu ?? -> Changed to latin N
        chars['??'] = 'N'  # Nu     ?? -> Changed to latin N
        chars['&Xi;'] = '??'  # Xi ??
        chars['&Omicron;'] = 'O'  # Omicron       ?? -> Changed to latin O
        chars['??'] = 'O'  # Omicron        ?? -> Changed to latin O
        chars['&Pi;'] = '??'  # Pi ??
        chars['&Rho;'] = 'P'  # Rho       ?? -> Changed to latin P
        chars['??'] = 'P'  # Rho    ?? -> Changed to latin P
        chars['&Sigma;'] = '??'  # Sigma   ??
        chars['&Tau;'] = 'T'  # Tau       ?? -> Changed to latin T
        chars['??'] = 'T'  # Tau    ?? -> Changed to latin T
        chars['&Upsilon;'] = 'Y'  # Upsilon       ?? -> Changed to latin Y
        chars['??'] = 'Y'  # Upsilon        ?? -> Changed to latin Y
        chars['&Phi;'] = '??'  # Phi       ??
        chars['&Chi;'] = 'X'  # Chi       ?? -> Changed to latin X
        chars['??'] = 'X'  # Chi    ?? -> Changed to latin X
        chars['&Psi;'] = '??'  # Psi       ??
        chars['&Omega;'] = '??'  # Omega   ??
        chars['&alpha;'] = 'a'  # alpha   ?? -> Changed to latin a
        chars['??'] = 'a'  # alpha  ?? -> Changed to latin a
        chars['&beta;'] = '??'  # beta     ??
        chars['&gamma;'] = '??'  # gamma   ??
        chars['&delta;'] = '??'  # delta   ??
        chars['&epsilon;'] = '??'  # epsilon       ??
        chars['&zeta;'] = '??'  # zeta     ??
        chars['&eta;'] = 'n'  # eta       ?? -> Changed to latin n
        chars['??'] = 'n'  # eta    ?? -> Changed to latin n
        chars['&theta;'] = '??'  # theta   ??
        chars['&iota;'] = '??'  # iota     ??
        chars['&kappa;'] = 'k'  # kappa   ?? -> Changed to latin k
        chars['??'] = 'k'  # kappa  ?? -> Changed to latin k
        chars['&lambda;'] = '??'  # lambda ??
        chars['&mu;'] = '??'  # mu ??
        chars['&nu;'] = 'v'  # nu ?? -> Changed to latin v
        chars['??'] = 'v'  # nu     ?? -> Changed to latin v
        chars['&xi;'] = '??'  # xi ??
        chars['&omicron;'] = 'o'  # omicron       ?? -> Changed to latin o
        chars['??'] = 'o'  # omicron        ?? -> Changed to latin o
        chars['&pi;'] = '??'  # pi ??
        chars['&rho;'] = 'p'  # rho       ?? -> Changed to latin p
        chars['??'] = 'p'  # rho    ?? -> Changed to latin p
        chars['&sigmaf;'] = '??'  # sigmaf ??
        chars['&sigma;'] = '??'  # sigma   ??
        chars['&tau;'] = 't'  # tau       ?? -> Changed to latin t
        chars['??'] = 't'  # tau    ?? -> Changed to latin t
        chars['&upsilon;'] = 'u'  # upsilon       ?? -> Changed to latin u
        chars['??'] = 'u'  # upsilon        ?? -> Changed to latin u
        chars['&phi;'] = '??'  # phi       ??
        chars['&chi;'] = '??'  # chi       ??
        chars['&psi;'] = '??'  # psi       ??
        chars['&omega;'] = '??'  # omega   ??
        chars['&thetasym;'] = '??'  # theta symbol ??
        chars['&upsih;'] = '??'  # upsilon symbol  ??
        chars['&piv;'] = '??'  # pi symbol ??
        chars['\u03BF'] = 'o'  # GREEK SMALL LETTER OMICRON
    else:
        # Greek Letters
        chars['&Alpha;'] = '??'  # Alpha
        chars['&Beta;'] = '??'  # Beta
        chars['&Gamma;'] = '??'  # Gamma   ??
        chars['&Delta;'] = '??'  # Delta   ??
        chars['&Epsilon;'] = '??'  # Epsilon
        chars['&Zeta;'] = '??'  # Zeta
        chars['&Eta;'] = '??'  # Eta
        chars['&Theta;'] = '??'  # Theta   ??
        chars['&Iota;'] = '??'  # Iota
        chars['&Kappa;'] = '??'  # Kappa
        chars['&Lambda;'] = '??'  # Lambda ??
        chars['&Mu;'] = '??'  # Mu
        chars['&Nu;'] = '??'  # Nu
        chars['&Xi;'] = '??'  # Xi
        chars['&Omicron;'] = '??'  # Omicron
        chars['&Pi;'] = '??'  # Pi ??
        chars['&Rho;'] = '??'  # Rho
        chars['&Sigma;'] = '??'  # Sigma   ??
        chars['&Tau;'] = '??'  # Tau
        chars['&Upsilon;'] = '??'  # Upsilon
        chars['&Phi;'] = '??'  # Phi       ??
        chars['&Chi;'] = '??'  # Chi
        chars['&Psi;'] = '??'  # Psi       ??
        chars['&Omega;'] = '??'  # Omega   ??
        chars['&alpha;'] = '??'  # alpha   ??
        chars['&beta;'] = '??'  # beta     ??
        chars['&gamma;'] = '??'  # gamma   ??
        chars['&delta;'] = '??'  # delta   ??
        chars['&epsilon;'] = '??'  # epsilon       ??
        chars['&zeta;'] = '??'  # zeta     ??
        chars['&eta;'] = '??'  # eta       ??
        chars['&theta;'] = '??'  # theta   ??
        chars['&iota;'] = '??'  # iota     ??
        chars['&kappa;'] = '??'  # kappa   ??
        chars['&lambda;'] = '??'  # lambda ??
        chars['&mu;'] = '??'  # mu ??
        chars['&nu;'] = '??'  # nu ??
        chars['&xi;'] = '??'  # xi ??
        chars['&omicron;'] = '??'  # omicron       ??
        chars['&pi;'] = '??'  # pi ??
        chars['&rho;'] = '??'  # rho       ??
        chars['&sigmaf;'] = '??'  # sigmaf ??
        chars['&sigma;'] = '??'  # sigma   ??
        chars['&tau;'] = '??'  # tau       ??
        chars['&upsilon;'] = '??'  # upsilon       ??
        chars['&phi;'] = '??'  # phi       ??
        chars['&chi;'] = '??'  # chi       ??
        chars['&psi;'] = '??'  # psi       ??
        chars['&omega;'] = '??'  # omega   ??
        chars['&thetasym;'] = '??'  # theta symbol ??
        chars['&upsih;'] = '??'  # upsilon symbol  ??
        chars['&piv;'] = '??'  # pi symbol ??
        chars['\u03BF'] = '??'  # GREEK SMALL LETTER OMICRON

    if lang.lower() != "ja":
        chars['\uFF5B'] = '{'  # ???
        chars['\uFF5D'] = '}'  # ???
        chars['\uFF08'] = '('  # ???
        chars['\uFF09'] = ')'  # ???
        chars['\uFF3B'] = '['  # ???
        chars['\uFF3D'] = ']'  # ???
        chars['\u3010'] = '('  # ???
        chars['\u3011'] = ')'  # ???
        chars['\u3002'] = '.'  # ???
        chars['\u3001'] = ','  # ???
        chars['\uFF0C'] = ','  # ???
        chars['\uFF1A'] = ':'  # ???
        chars['\uFF1B'] = ';'  # ???
        chars['\uFF1F'] = '?'  # ???
        chars['\uFF01'] = '!'  # ???
        chars['\uFF1C'] = '<'  # ???
        chars['\uFF1D'] = '='  # ???
        chars['\uFF1E'] = '>'  # ???
        chars['\uFF3F'] = '_'  # ???
        chars['\uFF40'] = "'"  # ???
    else:
        chars['{'] = '\uFF5B'  # ???
        chars['}'] = '\uFF5D'  # ???
        chars['('] = '\uFF08'  # ???
        chars[')'] = '\uFF09'  # ???
        chars['['] = '\uFF3B'  # ???
        chars[']'] = '\uFF3D'  # ???
        chars['\u3010'] = '\u3010'  # ??? -  #We maintain the same char
        chars['\u3011'] = '\u3011'  # ??? -  #We maintain the same char
        chars['\u3002'] = '\u3002'  # ??? #We maintain the same char
        chars[','] = '\u3001'  # ???
        chars[','] = '\uFF0C'  # ???
        chars['\u2026'] = '\u2026'  # ??? #We maintain the same char
        chars['\u2025'] = '\u2025'  # ??? #We maintain the same char
        chars[':'] = '\uFF1A'  # ???
        chars[';'] = '\uFF1B'  # ???
        chars['?'] = '\uFF1F'  # ???
        chars['!'] = '\uFF01'  # ???
        chars['<'] = '\uFF1C'  # ???
        chars['='] = '\uFF1D'  # ???
        chars['>'] = '\uFF1E'  # ???
        chars['_'] = '\uFF3F'  # ???
        chars["'"] = '\uFF40'  # ???

    charsRe = re.compile("(\\" + '|\\'.join(chars.keys()) + ")")

    return chars, charsRe


def getNormalizedPunctReplacements(lang):
    if lang.lower() == "fr":
        replacements = {
            " ,": ",",
            " )": ")",
            " }": "}",
            " ]": "]",
            #            " \""      :       "\"",          
            " ...": "...",

            "( ": "(",
            "{ ": "{",
            "[ ": "["
        }

    else:
        replacements = {
            " !": "!",
            " ?": "?",
            " :": ":",
            " ;": ";",
            " ,": ",",
            " )": ")",
            " }": "}",
            " ]": "]",
            #            " \""	:	"\"",
            " ...": "...",
            " ??": "??",

            "( ": "(",
            "{ ": "{",
            "[ ": "[",
            "?? ": "??",
            "?? ": "??"
        }
    rep = dict((re.escape(k), v) for k, v in replacements.items())
    pattern = re.compile("|".join(rep.keys()))
    return rep, pattern


# Orthographic corrections
def getReplacements(lang):
    replacements = {}
    input_replacements = None

    if lang.lower() in ["da", "de", "en", "es", "nb", "nl", "pt", "tr"]:
        input_replacements = open(os.path.dirname(os.path.realpath(__file__)) + "/replacements/replacements." + lang.lower(), "r")

    if input_replacements is not None:
        for i in input_replacements:
            field = i.split(u"\t")
            replacements[field[0].strip()] = field[1].strip()

    return replacements


def replace_chars(match):
    global global_chars_lang
    char = match.group(0)
    return global_chars_lang[char]


def replace_chars3(match):
    char = match.group(0)
    return ""


def fix(text, lang, chars_rep, chars_pattern, punct_rep, punct_pattern):
    global global_chars_lang
    global_chars_lang = chars_rep

    # htmlEntity=regex.compile(r'[&][[:space:]]*[#][[:space:]]*[0-9]{2,4}[[:space:]]*[;]?',regex.U)
    chars3Re = regex.compile("[\uE000-\uFFFF]")
    chars3Re2 = regex.compile("[\u2000-\u200F]")
    chars3Re3 = regex.compile("\u007F|[\u0080-\u00A0]")
    quotesRegex = regex.compile("(?P<start>[[:alpha:]])\'\'(?P<end>(s|S|t|T|m|M|d|D|re|RE|ll|LL|ve|VE|em|EM)\W)")
    collapse_spaced_entities = regex.compile('([&][ ]*[#][ ]*)([0-9]{2,6})([ ]*[;])')

    stripped_text = re.sub(' +', ' ', text.strip()).strip(" \n")  # Collapse multiple spaces
    collapsed_entities = collapse_spaced_entities.sub("&#\\2;", stripped_text)

    # Test encode: fix mojibake
    ftfy_fixed_text = " ".join([ftfy.fix_text_segment(word, fix_entities=True, uncurl_quotes=False, fix_latin_ligatures=False) for word in collapsed_entities.split()])
    # ftfy_fixed_text= ftfy.fix_text_segment(stripped_text, fix_entities=True,uncurl_quotes=False,fix_latin_ligatures=False)

    # nicely_encoded_text = htmlEntity.sub(html.unescape, nicely_encoded_text)
    nicely_encoded_text = html.unescape(ftfy_fixed_text)

    # First replacing all HTML entities
    # for substring in htmlEntity.findall(nicely_encoded_text):
    #    code=substring.replace(' ','')[2:].replace(';','')
    #    try:
    #        newChar=chr(int(code))
    #    except ValueError:
    #        newChar=code    
    #    if newChar != "\n":
    #        nicely_encoded_text = nicely_encoded_text.replace(substring,newChar)

    normalized_text = chars_pattern.sub(replace_chars, nicely_encoded_text)

    if lang.lower() != "ja":
        normalized_text = chars3Re.sub(replace_chars3, normalized_text)
    normalized_text = chars3Re2.sub(replace_chars3, normalized_text)
    normalized_text = chars3Re3.sub(replace_chars3, normalized_text)
    normalized_text = quotesRegex.sub("\g<start>\'\g<end>", normalized_text)
    normalized_text_with_normalized_punct = punct_pattern.sub(lambda m: punct_rep[re.escape(m.group(0))], normalized_text)

    return normalized_text_with_normalized_punct.strip()


def orthofix(text, replacements):
    if len(replacements) > 0:
        last = 0
        line = []

        for j in regex.finditer(r"([^-'[:alpha:]](?:[^-[:alpha:]']*[^-'[:alpha:]])?)", text):
            if last != j.start():
                line.append((text[last:j.start()], "w"))
            line.append((text[j.start():j.end()], "s"))
            last = j.end()
        else:
            if last != len(text):
                line.append((text[last:], "w"))
        fixed_text = ""
        for j in line:
            if j[1] == "w":
                if j[0] in replacements:
                    fixed_text += replacements[j[0]]
                else:
                    fixed_text += j[0]
            else:
                fixed_text += j[0]
    else:
        fixed_text = text

    return fixed_text
