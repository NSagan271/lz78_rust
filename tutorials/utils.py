def pretty_print(generated, max_line_width=80):
    for line in generated.split("\n"):
        line_width = 0
        for line2 in line.split("\t"):
            for word in line2.split(" "):
                if line_width + len(word) + 1 > max_line_width:
                    line_width = len(word)
                    print()
                    while line_width > max_line_width:
                        line_width -= max_line_width
                        print(word[:max_line_width])
                        word = word[max_line_width]                        
                else:
                    line_width += len(word)
                print(word, end=" ")
            if line_width + 8 <= max_line_width:
                line_width += 8
                print("\t", end="")
        print()