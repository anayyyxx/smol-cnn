import string

chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
