import psutil
 
for proc in psutil.process_iter():
    print(proc.name())
 
def kill_by_process_name(name):
    for proc in psutil.process_iter():
        if proc.name() == name:
            print("Killing process: " + name)
            if(check_process_exist_by_name(name)):
                print("Killing process: " + name + " sucess")
            else:
                print("Killing process: " + name + " failed")
            return
 
    print("Not found process: " + name)
 
def check_process_exist_by_name(name):
    for proc in psutil.process_iter():
        if proc.name() == name:
            return True
 
    return False
 
kill_by_process_name("iTunesHelper.exe")   