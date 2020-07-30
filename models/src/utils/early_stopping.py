class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_acc = None

    def __call__(self, score):
        if self.best_acc is None:
            self.best_acc = score
            return False
        elif score < self.best_acc:
            self.counter += 1
            if self.patience <= self.counter:
                print(f'Early Stopping: Accuracy only decreased over last {self.patience} Epochs')
                return True
        else:
            self.best_acc = score
            self.counter = 0
            return False
