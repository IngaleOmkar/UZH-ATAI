class FormatHelper:
    @staticmethod
    def array_to_sentence(arr):
        if not arr:
            return ""
        elif len(arr) == 1:
            return arr[0]
        elif len(arr) == 2:
            return " and ".join(arr)
        else:
            return ", ".join(arr[:-1]) + ", and " + arr[-1]