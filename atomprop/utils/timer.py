import time

class TrainingTimer:
    def __init__(self):
        self.start_time = time.time()
    
    def elapsed(self):
        return time.time() - self.start_time
    
    def format_elapsed(self):
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.2f} second"
        elif elapsed < 3600:
            return f"{elapsed / 60:.2f} min"
        else:
            return f"{elapsed / 3600:.2f} hour"

    def print(self, *args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        print(f"[{self.format_elapsed()}] {msg}", **kwargs)
