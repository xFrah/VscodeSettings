import os
import shutil
import subprocess


def clean():
    for root, dirs, files in os.walk("build"):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def change_file(idd: int, tag: bool, beam: bool):
    folder = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(folder, "ex_05c_main.c")
    with open(file_path, "r") as file:
        data = file.readlines()
    success1, success2, success3 = False, False, False
    for i in range(len(data)):
        if "uint16_t static const BEAM_DEVICE_ID" in data[i]:
            data[i] = "uint16_t static const BEAM_DEVICE_ID = %s;\n" % idd
            success1 = True
        if "#define BEAMDIGITAL_BOARD 1" in data[i]:
            data[i] = f"{'' if beam else '//'}#define BEAMDIGITAL_BOARD 1\n"
            success2 = True
        if "int k =" in data[i]:
            data[i] = f"int k = {'0' if tag else '2'};\n"
            success3 = True
    with open(file_path, "w") as file:
        file.writelines(data)
    if not success1:
        print("Error: Could not find ID row in file: %s" % file_path)
    if not success2:
        print("Error: Could not find board row in file: %s" % file_path)
    if not success3:
        print("Error: Could not find k row in file: %s" % file_path)


def compiler(idd, tag=True, beam=False):
    clean()
    change_file(idd, tag=tag, beam=beam)
    subprocess.run('cd build && cmake -G "Unix Makefiles" .. && make', capture_output=True)
    folder = os.path.dirname(os.path.realpath(__file__))
    hex_path = os.path.join(folder, "build", "zephyr", "zephyr.hex")
    new_hex_path = os.path.join(folder, f"zephyr_{idd}.hex")
    shutil.copyfile(hex_path, new_hex_path)

if __name__ == "__main__":
    files = [
        (1, True, False),
        (2, True, False),
        (3, True, False),
    ]
    for file in files:
        compiler(*file)
