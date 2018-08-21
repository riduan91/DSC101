# -*- coding: utf-8 -*-

# Thư mục chứa các hình ảnh gốc sẽ được sử dụng
RAW_IMAGE_FOLDER = "RawData/face/format2"

# Thư mục chứa các hình ảnh được copy, tập trung và đổi tên
IMAGE_FOLDER = "ImageFolder"

# Thư mục chứa các file text lưu trích xuất khuôn mặt, đôi mắt dưới dạng vector số, output của phần 1
TEXT_DATA_FOLDER = "TextData"

# Định dạng hình ảnh
EXTENSION = ".png"

# File Haar cascade của khuôn mặt
HAARCASCADE_FRONTALFACE_DEFAULT = "Configuration/haarcascade_frontalface_default.xml"

# File Haar cascade của đôi mắt
HAARCASCADE_EYE_DEFAULT = "Configuration/haarcascade_eye.xml"

# Hằng số kiểm tra xem khuôn mặt có được chụp ở tư thế trực diện không
# Khuôn mặt được xem là có tư thế trực diện nếu tâm mắt trái và tâm mắt phải có tung độ không cách nhau quá 3 pixel
HORIZONTAL_CHECK = 3

VERTICAL_CHECK = 10

# File text lưu trích xuất khuôn mặt, đôi mắt dưới dạng vector số, output của phần 1
FACES_DATA = "TextData/Faces.csv"
EYES_DATA = "TextData/Eyes.csv"

# Chuẩn hoá tên các ca sĩ từ thư mục hình ảnh gốc
SINGER_NAME_DICTIONARY = {
    "bao thy" : "BaoThy",
    "chi pu": "ChiPu",
    "dam vinh hung": "DamVinhHung",
    "dan truong": "DanTruong",
    "ha anh tuan": "HaAnhTuan",
    "ho ngoc ha": "HoNgocHa",
    "huong tram": "HuongTram",
    "lam truong": "LamTruong",
    "my tam": "MyTam",
    "No phuoc thing": "NooPhuocThinh",
    "son tung": "SonTung",
    "tuan hung": "TuanHung"
}

# Đánh số thứ tự tên các ca sĩ
SINGER_INDEX_DICTIONARY = {
    "BaoThy": 0,
    "ChiPu": 1,
    "DamVinhHung": 2,
    "DanTruong": 3,
    "HaAnhTuan": 4,
    "HoNgocHa": 5,
    "HuongTram": 6,
    "LamTruong": 7,
    "MyTam": 8,
    "NooPhuocThinh": 9,
    "SonTung": 10,
    "TuanHung": 11
}

# Sau khi xử lí các hình ảnh và sắp xếp theo thứ tự ABC, các hình ảnh có thứ tự 0 đến 220 sẽ thuộc ca sĩ BaoThy
# v.v.
SINGER_IMAGE_RANGE = {
    "BaoThy": range(0, 221),
    "ChiPu": range(221, 521),
    "DamVinhHung": range(521, 676),
    "DanTruong": range(676, 811),
    "HaAnhTuan": range(811, 921),
    "HoNgocHa": range(921, 1109),
    "HuongTram": range(1109, 1327),
    "LamTruong": range(1327, 1407),
    "MyTam": range(1407, 1580),
    "NooPhuocThinh": range(1580, 1820),
    "SonTung": range(1820, 2020),
    "TuanHung": range(2020, 2226)
}
    