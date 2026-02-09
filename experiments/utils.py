def add_book_figure_name(book_figure_name, plot_name):
    name = plot_name

    if book_figure_name is not None:
        name = f'{book_figure_name}: ' + name
    
    return name

def print_separator_line():
    print("="*40)