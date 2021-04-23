import os

def paste_queries(input_fp, query_fp, output_fp, delim=' ||| '):
    print(f"Pasting {input_fp} {query_fp} -> {output_fp} ...")
    with open(input_fp, 'r', encoding='utf-8') as input_fh, \
         open(query_fp, 'r', encoding='utf-8') as query_fh, \
         open(output_fp, 'w', encoding='utf-8') as output_fh:
        for input_line in input_fh:
            input_line = input_line.strip()
            query_line = query_fh.readline().strip()
            if query_line:
                output_line = f"{input_line}{delim}{query_line}\n"
            else:
                output_line = f"{input_line}\n"
            output_fh.write(output_line)

if __name__ == '__main__':
    import fire
    fire.Fire(paste_queries)