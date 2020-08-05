class RawData(object):
    def __init__(self, datetime, open, close, high, low):
        self.datetime = datetime
        self.open = open
        self.close = close
        self.high = high
        self.low = low


def read_sample_data(path):
    print("past records check...")
    raw_data = []
    separator = ","
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("datetime"): 
                continue
            l = line[:-1]
            fields = l.split(separator)
            if len(fields) > 4:
                raw_data.append(RawData(str(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])))
    # sorted_data = sorted(raw_data, key=lambda x: x.datetime)
    sorted_data = raw_data
    print("got %s records." % len(sorted_data))
    print(sorted_data)
    return sorted_data