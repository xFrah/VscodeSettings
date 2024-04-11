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
            newline = "uint16_t static const BEAM_DEVICE_ID = %s;\n" % idd
            print(f"Changing ID from '{data[i]}' to '{newline}'")
            data[i] = newline
            success1 = True
        if "#define BEAMDIGITAL_BOARD 1" in data[i]:
            newline = f"{'' if beam else '//'}#define BEAMDIGITAL_BOARD 1\n"
            print(f"Changing board from '{data[i]}' to '{newline}'")
            data[i] = newline
            success2 = True
        if "int k =" in data[i]:
            newline = f"int k = {'0' if tag else '2'};\n"
            print(f"Changing k from '{data[i]}' to '{newline}'")
            data[i] = newline
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
    folder = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(folder, "build")
    cmd = f'cd {build_path} && cmake -G "Unix Makefiles" .. && make'
    os.system(cmd)
    hex_path = os.path.join(folder, "build", "zephyr", "zephyr.hex")
    new_hex_path = os.path.join(folder, f"zephyr_{idd}_{'beam' if beam else 'dwm'}_{'tag' if tag else 'danger'}.hex")
    shutil.copyfile(hex_path, new_hex_path)
    clean()


if __name__ == "__main__":
    # id, tag, beam
    files = [
        (1, False, False),
    ]
    for file in files:
        compiler(*file)
