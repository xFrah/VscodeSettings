import os
import shutil


def clean():
    for root, dirs, files in os.walk("build"):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def change_file(idd, tag=True, beam=False):
    folder = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(folder, "ex_05c_main.c")
    with open(file_path, "r") as file:
        data = file.readlines()
    success1, success2 = False, False
    for i in range(len(data)):
        if "uint16_t static const BEAM_DEVICE_ID" in data[i]:
            data[i] = "uint16_t static const BEAM_DEVICE_ID = %s;\n" % idd
            with open(file_path, "w") as file:
                file.writelines(data)
            success1 = True
        if "#define BEAMDIGITAL_BOARD 1" in data[i]:
            if tag:
                data[i] = "#define BEAMDIGITAL_BOARD 1\n"
            else:
                data[i] = "// #define BEAMDIGITAL_BOARD 1\n"
            with open(file_path, "w") as file:
                file.writelines(data)
            success2 = True
    if not success:
        print("Error: Could not find line to change in file: %s" % file_path)


def compiler(id, tag=True, beam=False):
    clean()
    os.system('cd build && cmake -G "Unix Makefiles" .. && make')


if __name__ == "__main__":
    change_row32(3)
