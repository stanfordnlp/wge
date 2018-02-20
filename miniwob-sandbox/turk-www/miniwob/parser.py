import re

classes = set()
with open("logfile", "r") as f:
    for line in f:
        match = re.match(r".*class='([^']+)'", line)
        if match:
            classes.add(match.group(1))
        else:
            match = re.match(r".*class=\"([^\"]+)\"", line)
            if match:
                classes.add(match.group(1))
            else:
                print line,

#for c in classes:
#    print "\"{}\",".format(c)
