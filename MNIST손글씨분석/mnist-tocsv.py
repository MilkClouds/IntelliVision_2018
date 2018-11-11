import struct
def to_csv(name, maxdata):
    # 레이블 파일과 이미지 파일 열기
    label_file = open("./mnist/"+name+"-labels-idx1-ubyte", "rb")
    img_file = open("./mnist/"+name+"-images-idx3-ubyte", "rb")
    csv_f = open("./mnist/"+name+".csv", "w", encoding="utf-8")
    # 헤더 정보 읽기 (★1)
    mag, lbl_count = struct.unpack(">II", label_file.read(8))
    mag, img_count = struct.unpack(">II", img_file.read(8))
    rows, cols = struct.unpack(">II", img_file.read(8))
    pixels = rows * cols
    print(lbl_count,img_count)
    # 이미지 데이터를 읽고 CSV로 저장하기 (★2)
    res = []
    for idx in range(lbl_count):
        if idx > maxdata: break
        label = struct.unpack("B", label_file.read(1))[0]
        bdata = img_file.read(pixels)
        sdata = list(map(lambda n: str(n), bdata))
        csv_f.write(str(label)+",")
        csv_f.write(",".join(sdata)+"\r\n")
    csv_f.close()
    label_file.close()
    img_file.close()

to_csv("train", 30000)
to_csv("t10k", 500)