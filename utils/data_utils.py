r"""A set of data utilities for CUB dataset.

Provides a utility for reading a dataset of the form:
    folder/
        class0/item0.txt
        class0/item1.txt
        class1/item0.txt ..
    OR
    folder/
        class0/item0.jpg ..

Returns a list of filenames in the dataset based on specifications.

Author: Ramakrishna Vedantam
Email: vrama91@vt.edu
Organization: Computer Vision Lab, Virginia Tech
"""
import random
from nltk import word_tokenize

import tensorflow as tf

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def find_files_from_hierarchy(data_dir,
                              labels_file,
                              input_split_file,
                              split_name,
                              pattern='.jpg',
                              permute=True):
    r"""Build a list of all images files and labels in the data set.

    Args:
        data_dir: string, path to the root directory of images.

        Assumes that the image data set resides in JPEG files located in
        the following directory structure.

            data_dir/dog/another-image.jpg
            data_dir/dog/my-image.jpg

        where 'dog' is the label associated with these images.

        labels_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
            dog
            cat
            flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.

        input_split_file: string, path to the input split file.

            The list of images along with the split they belong to is held in
            this
            file. Each line corresponds to the image name and the split name.
            For example:
                001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg\
                    Validation

        split_name: string, name of the split we are interested in processing.
            Must be one of "Train", "Validation", "Test"

        pattern: A pattern to use for filtering files. For example, if
            searching for only JPEG images, the pattern can be '.jpg'

        permute: True/False, indicates whether to permute the file list or not.

    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip()
                     for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    datum_to_split = {
        l.strip().split(' ')[0]: l.strip().split(' ')[1]
        for l in tf.gfile.FastGFile(input_split_file, 'r').readlines()
    }

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        file_path = ('%s/%s/*' + pattern) % (data_dir, text)
        matching_files = tf.gfile.Glob(file_path)
        # Filter matching files based on the desired split, sort filenames to
        # get a consistent ordering.
        # TODO(vrama): Make the data split file without txt, jpg extensions.
        # TODO(vrama): This is really brittle, '.jpg'
        matching_files = sorted([
            matching_file for matching_file in matching_files
            if datum_to_split['/'.join(matching_file.split('/')[-2:]).replace(
                pattern, '.jpg')] == split_name
        ])

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' %
                  (label_index, len(unique_labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    if permute:
        print("inside permute")
        random.seed(12345)
        random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    pattern_string = pattern
    if pattern_string == '':
        pattern_string = '*'

    print('Found %d files matching pattern %s across %d labels inside %s.' %
          (len(filenames), pattern_string, len(unique_labels), data_dir))
    return filenames, texts, labels


def shuffle_entries(*lists):
    """Shuffle entries in a set of lists whose corresponding entries align.

    Given a collection of lists, where lists[1][i], lists[2][i] ... all are
    aligned with each other, permute the lists to create another list with the
    alignment preserved.
    Args:
        lists: a tuple of lists
    Returns:
        permuted_lists: a tuple of permuted lists, with alignment preserved.
    """
    # All lists should have same number of elements.
    _lengths = (len(lists) for list_element in lists)
    assert (_lengths.count(_lengths[0]) == len(_lengths))

    shuffled_index = list(range(_lengths[0]))
    random.seed(12345)
    random.shuffle(shuffled_index)

    output_permuted_lists = []
    for _list in lists:
        _list = [_list[i] for i in shuffled_index]
        output_permuted_lists.append(_list)


def _is_png(filename):
    """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def preprocess_sentence(sentence):
    """Add a full stop to each sentence and make it lower case."""
    return sentence.rstrip('\n').rstrip('.').lower() + '.'


def _process_sentence(sentence, vocabulary):
    """Denseley-encode a sentence using a given vocabulary.

    Given as input a lower case sentence first tokenize it, and prepend it with
    start of sentence token, and append it with end of sentence token. Replace
    any words non-existent in the vocabulary with the unknown word token.

    Args:
        sentence: string, containing the sentence to process.
        vocabulary: an object of the vocabulary class.
    Returns:
        processed_index: list, a list of int64 encodings of the sentence using
            the vocabulary.
        processed_tokens: list, a list of tokens in the sentence.
    """
    tokenized_sentence = (
        [vocabulary.start] + word_tokenize(sentence) + [vocabulary.end])

    processed_index = []
    processed_tokens = []
    for word in tokenized_sentence:
        use_word = word
        if word not in vocabulary.vocab:
            use_word = vocabulary.unk
        processed_tokens.append(use_word)
        processed_index.append(vocabulary.word_to_idx[use_word])

    return processed_index, processed_tokens


def _is_split_valid(split_name):
    return split_name in ["Train", "Validation", "Test"]


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
