
def make_filename(filename_base=None, **kwargs):
    filename = [filename_base]
    for key, value in kwargs.items():
        filename.append(f'{key}-{value}')
    return "_".join(filename)
