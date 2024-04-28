import argparse
import os
import csv
from data_utils import *
from summary_utils import *
from image_utils import *


def process_books(books, summarizer, text_to_image):

    if not os.path.exists('./book_covers'):
        os.makedirs('./book_covers')

    if len(books)>1:
        csv_path = 'summarized_books.csv'
    else:
        csv_path = f"summarized_book_{books.index[0]}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    columns = [x for x in books.columns] + ['keywords', 'plot', 'cover_path']
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for index, book in books.iterrows():
            book['keywords'] = keyword_extraction(book['summary'])
            book['plot'] = summarizer.summarize(book['summary'])
            image = text_to_image.transform(book['plot'])
            image_path = f"./book_covers/{index}_{book['book_title']}.png"
            image.save(image_path)
            book['cover_path'] = image_path
            writer.writerow(book)

def main(datapath, book_title, sum_model_name, sum_model_path, cuda):
    all_books = create_dataset(datapath)

    if book_title is not None:
        book_title_lower = book_title.lower()
        book_rows = all_books[all_books['book_title'].str.lower() == book_title_lower]
        if book_rows.empty:
            print('The book is NOT included!')
            return
    else:
        print("You Are Processing ALL the Books!")
        analyze_dataset(all_books)
        book_rows = all_books

    if sum_model_name == "lexrank":
        summarizer = LexRankSumm()
    elif sum_model_name.startswith("t5"):
        summarizer = T5Summ(sum_model_name, sum_model_path, cuda)
    else:
        print('Invalid Summarization Model!')
        return

    text_to_image = StableDiff()

    process_books(book_rows, summarizer, text_to_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for Data Path and Models")
    parser.add_argument("--datapath", type=str, default='booksummaries.txt', help="Path to the data")
    parser.add_argument("--book_title", type=str, default=None, help="The title of the book")
    parser.add_argument("--sum_model_name", type=str, default='lexrank', help="Name a T5 Model (If not provided, LexRank; an extractive method)")
    parser.add_argument("--sum_model_path", type=str, default=None, help="Path to summarization model (In case you have a local pretrained model)")
    parser.add_argument("--cuda", type=bool, default=True, help="Do You Want to Use CUDA? Write True or False. Default is True.")
    args = parser.parse_args()

    main(args.datapath, args.book_title, args.sum_model_name, args.sum_model_path, args.cuda)
