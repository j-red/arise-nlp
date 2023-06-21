import datetime as dt
from timefhuman import timefhuman
import time, json

colors = { # useful for printing colorful debug messages
    "white": '\033[0m',
    "red": '\033[31m',
    "green": '\033[32m',
    "orange": '\033[33m',
    "blue": '\033[34m', 
    "purple": '\033[35m'
}

# "key": ("start", "end", referenceTime) # referenceTime is dt.now() if not set
overrides = {
    'last weekend': ('last saturday', 'last monday at midnight', None),
    'last week': ('last monday at midnight', 'midnight', "last saturday"), # monday at midnight .. saturday midnight
    'past week': ('7 days ago', dt.datetime.now(), None), 
    'last 24 hours': (dt.datetime.now() - dt.timedelta(hours=24), dt.datetime.now(), None),
    'recently': ('7 days ago', dt.datetime.now(), None), # `recently` defaults to past 7 days
    'yesterday': ('yesterday at midnight', 'today at midnight', None),
    'this morning': ('today at 6 am', 'today at noon', None),
    'this afternoon': ('today at noon', 'today at 6pm', None),
    'last night': ('yesterday at 6 pm', 'today at midnight', None),
}


def str_to_datetime(x : str, reference=dt.datetime.now(), ref_end=dt.datetime.now()):
    """ Convert relative query strings into concrete datetime windows.
        e.g; x="the day before", reference=<datetime object for yesterday> -> <datetime object for two days ago>
    """
    
    def str2int(textnum):
        """ Helper function to convert natural language quantities into integers. """
        numwords = {}
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

        current = result = 0
        for word in textnum.split():
            if word not in numwords:
                # raise Exception("Illegal word: " + word)
                continue

            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current
    
    
    parts = x.lower().split()
    print(f"\nParts: {parts}\n\n")
    
    if "before" in parts or "prior" in parts or "previous" in parts:  # if query string contains indicators of a backwards time offset
        if "hour" in parts: # 'the hour before' 
            start = reference - dt.timedelta(hours=1) 
            end = reference
        elif "day" in parts: # 'the day before'
            start = reference - dt.timedelta(hours=24)
            start = start.replace(hour=0, minute=0, second=0) # midnight the day prior
            end = reference.replace(hour=0, minute=0, second=0) # midnight the current day
        elif "days" in parts: # 'two days before'
            
            offset = 24 * str2int(x.lower())
            print('Days Offset:', str(offset))
            
            end = start.replace(hour=0, minute=0, second=0) # midnight the current day
            start = reference - dt.timedelta(hours=offset)
            start = start.replace(hour=0, minute=0, second=0) # midnight the day prior
        elif "week" in parts: # 'the week before'
            _ref = timefhuman("last saturday", now=reference)
            start = timefhuman("last saturday at midnight", now=_ref)
            end = timefhuman("last monday at midnight", now=_ref)
        elif "month" in parts: # 'the month before'
            start = reference - dt.timedelta(days=30)
            start = start.replace(day=1, hour=0, minute=0, second=0) # midnight on first day of prev month
            end = reference.replace(day=1, hour=0, minute=0, second=0) # midnight on first day of current month
    
    elif "after" in parts or "following" in parts or "subsequent" in parts or "next" in parts:
        reference = ref_end # reference the end of the current window instead of the beginning
        if "hour" in parts: # 'the hour after' 
            start = reference
            end = reference + dt.timedelta(hours=1) 
        elif "day" in parts: # 'the next day'
            start = reference + dt.timedelta(hours=24)
            start = start.replace(hour=0, minute=0, second=0) # midnight the next day 
            end = start + dt.timedelta(hours=24)
            end = end.replace(hour=0, minute=0, second=0) # midnight at the end of the next day
        elif "days" in parts: # 'two days after'
            start = reference + dt.timedelta(hours=24)
            start = start.replace(hour=0, minute=0, second=0) # midnight the next day
            
            offset = 24 * str2int(x.lower())
            end = reference + dt.timedelta(hours=offset)
            end = end.replace(hour=0, minute=0, second=0)
        elif "week" in parts: # 'the week after'
            _ref = timefhuman("last saturday", now=reference)
            start = timefhuman("last saturday at midnight", now=_ref)
            end = timefhuman("last monday at midnight", now=_ref)
        elif "month" in parts: # 'the subsequent month'
            start = reference 
            end = reference + dt.timedelta(days=30)
            end = end.replace(hour=0, minute=0, second=0)
        elif "months" in parts:
            offset = 30 * str2int(x.lower()) # turn natural language (e.g., 'two', 'three' into ints)
            start = reference 
            end = reference + dt.timedelta(days=30 * offset)
            end = end.replace(hour=0, minute=0, second=0)
    
    # print(f"New Start: {start}, End: {end}")
    return (start, end)

class ClassificationTask:
    def __init__(self, subtasks: list, start="", end="", ref=dt.datetime.now()):
        # Start and End times should be in datetime format (or Unix timestamp : int)
        # e.g., start="2019-01-01 12:50:15", end=1332938930
        
        self.tasks = subtasks
        
        if str(start).lower() in overrides.keys():
            print(f'Overriding `{start}` to `{overrides[start]}`')
            start, end, ref = overrides[str(start)]
        else:
            # print(start, "not in overrides")
            pass
        
        if type(ref) == str:
            try:
                ref = timefhuman(ref) # attempt to parse input
            except:
                print(f"{colors['red']}Error parsing reference time `{ref}`.{colors['white']}")
        elif type(ref) != dt.datetime:
            ref = dt.datetime.now() # default to current time if not presented as override
            
        self.start = self.parse(start, reference=ref)
        self.end = self.parse(end, fallback=dt.datetime.now(), reference=ref)
        
        if self.start > self.end: # swap start and end times
            print(f"{colors['red']}Error: Query window ends before it begins. Swapping start and end times.{colors['white']}")
            _ = self.start
            self.start = self.end
            self.end = _
        
        print(f"Start Time: {self.start}, End Time: {self.end}")
        print(f"Window Length: {self.end-self.start}")
        print(f"Tasks: {self.tasks}")
        
        return
    
    def parse(self, time, format_string="%Y-%m-%d %H:%M:%S", fallback=dt.datetime.utcfromtimestamp(0), reference=dt.datetime.now()):
        if type(time) == int:
            return dt.datetime.utcfromtimestamp(time) # convert from Unix timestamp to UTC datetime
        if type(time) == dt.datetime:
            return time # return input datetime object
        if type(time) == str: 
            try:
                value = dt.datetime.strptime(start, format_string) # convert input string to datetime
                return value
            except:
                value = timefhuman(time, now=reference)
                if value:
                    return value # don't return value if cannot parse
        return fallback
    
    def jsonify(self) -> str:
        _json = {
            "Subtasks": self.tasks,
            "Start": round(time.mktime(self.start.timetuple())), # create Unix timestamp from UTC datetime objects
            "End": round(time.mktime(self.end.timetuple()))
        }
        self.json = json.dumps(_json, sort_keys=False, indent=4)
        return self.json
    
    def as_dict(self) -> dict:
        _self = {
            "Subtasks": self.tasks,
            "Start": round(time.mktime(self.start.timetuple())), # create Unix timestamp from UTC datetime objects
            "End": round(time.mktime(self.end.timetuple()))
        }
        return _self
    
    def __repr__(self):
        return f"CLASSIFICATION QUERY: \n\tSubtasks: {self.tasks}\n\tStart:\t{self.start}\n\tEnd:\t{self.end}"

class FindQuery(ClassificationTask):
    def __repr__(self):
        return f"FIND QUERY: \n\tSubtasks: {self.tasks}"

class InferQuery(ClassificationTask):
    def __repr__(self):
        return f"INFER QUERY: \n\tSubtasks: {self.tasks}"


def task_from_dict(d):
    return ClassificationTask(d['Subtasks'], start=d['Start'] if 'Start' in d else "", end=d['End'] if 'End' in d else "")