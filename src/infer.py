import argparse

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type = str, default = 'mean.th', help='model, default mean.th')
    parser.add_argument('input_file', type = str, default = 'input.txt', help='input file, default input.txt')
    parser.add_argument('output_file', type = str, default = 'output.txt', help='output file, default output.txt')

# def infer():
flags = parse_flags()
flag_keys = ['checkpoint_path', 'input_file', 'output_file']
(checkpoint_path, input_file, output_file) = itemgetter(*flag_keys)(vars(flags))

# if __name__ == '__main__':
#     infer()
