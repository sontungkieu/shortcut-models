"""Imagenet 2012 Streaming from Hugging Face custom repo."""

import tensorflow as tf
import tensorflow_datasets as tfds

# CẤU HÌNH REPO CỦA BẠN
HF_REPO = "codemaivanngu/imagenet-tfrecord"
HF_BRANCH = "main"
NUM_SHARDS_TRAIN = 641  # 1.28M ảnh / 2000 ảnh mỗi file ~ 641 files
NUM_SHARDS_VAL = 26     # 50k ảnh / 2000 ảnh mỗi file ~ 25-26 files

class Imagenet2012(tfds.core.GeneratorBasedBuilder):
    """Custom ImageNet builder that streams from Hugging Face."""

    VERSION = tfds.core.Version('5.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="ImageNet 2012 streamed from Hugging Face TFRecords.",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(encoding_format='jpeg'),
                'label': tfds.features.ClassLabel(num_classes=1000),
                'file_name': tfds.features.Text(), # Giữ field này để match với kvfrans
            }),
            supervised_keys=('image', 'label'),
            homepage='https://huggingface.co/datasets/' + HF_REPO,
        )

    def _split_generators(self, dl_manager):
        # Chúng ta không download gì cả, chỉ khai báo các split
        return {
            'train': self._generate_examples('train'),
            'validation': self._generate_examples('validation'),
        }

    def _generate_examples(self, split):
        # Hàm này bắt buộc phải có nhưng sẽ không được dùng 
        # vì ta sẽ override _as_dataset bên dưới để stream.
        return []

    def _as_dataset(self, split=tfds.Split.TRAIN, decoders=None, read_config=None, shuffle_files=None):
        """
        Đây là trái tim của kỹ thuật Streaming.
        Thay vì đọc từ đĩa cứng (TFDS format), ta trả về Dataset đọc từ URL (HF format).
        """
        
        # 1. Tạo danh sách URL trỏ thẳng vào file trên Hugging Face
        if split == 'train':
            split_name = 'train'
            num_shards = NUM_SHARDS_TRAIN
        else:
            split_name = 'validation'
            num_shards = NUM_SHARDS_VAL

        # Link format: https://huggingface.co/datasets/{USER}/{REPO}/resolve/{BRANCH}/data/{split}/{filename}
        base_url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_BRANCH}/data/{split_name}"
        
        # Tạo list các URL
        file_urls = [
            f"{base_url}/{split_name}-{i:05d}.tfrecord" 
            for i in range(num_shards)
        ]

        # 2. Tạo TFRecordDataset từ URL
        # Quan trọng: compression_type='GZIP' vì dataset của bạn đã nén
        if read_config and read_config.shuffle_seed:
             # Nếu TFDS yêu cầu shuffle file (ví dụ lúc train)
             import random
             random.seed(read_config.shuffle_seed)
             random.shuffle(file_urls)

        # Sử dụng num_parallel_reads=AUTOTUNE để download đa luồng
        dataset = tf.data.TFRecordDataset(
            file_urls, 
            compression_type='GZIP', 
            num_parallel_reads=tf.data.AUTOTUNE
        )

        # 3. Hàm giải mã (Parser)
        # Biến đổi từ Raw Bytes của bạn -> Tensor (H, W, 3) mà tfds.load mong đợi
        def parse_hf_record(example_proto):
            feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            
            # Decode JPEG thành Tensor uint8 (để khớp behavior chuẩn của tfds.load)
            image = tf.io.decode_jpeg(parsed['image'], channels=3)
            
            # Trả về dict đúng cấu trúc features khai báo trong _info
            return {
                'image': image,
                'label': parsed['label'],
                'file_name': tf.constant(f"{split_name}.tfrecord") # Dummy filename
            }

        # 4. Map parser vào dataset
        dataset = dataset.map(parse_hf_record, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset