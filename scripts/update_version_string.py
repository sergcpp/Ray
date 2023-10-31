
import re
import subprocess
import sys

def main():
    version_string = subprocess.check_output(['git', 'describe', '--dirty', '--always', '--tags'])
    file_name = sys.argv[1]

    all_lines = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            m = re.search(r'^const char \*(\w+)::Version\(\)\s*(const)?\s*\{ return ".*"; \}', line)
            if m:
                NS = str(m[1])
                line = 'const char *' + NS + '::Version() ';
                if m[2] != None:
                    line += 'const '
                line += '{ return "'
                line += version_string.decode('UTF-8')[0:-1]
                line += '"; }\n'
                print('Writing version: ', version_string.decode('UTF-8')[0:-1])
            all_lines.append(line)

    # rewrite file
    with open(file_name, "w", encoding='utf-8') as f:
        for line in all_lines:
            f.write(line)

if __name__ == "__main__":
    main()
