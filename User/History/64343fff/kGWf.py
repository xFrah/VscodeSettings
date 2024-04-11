import requests


def main():
    # make request to 5.196.23.212/killswitch
    try:
        res = requests.get("http://5.196.23.212/killswitch", timeout=5)
        if res.status_code == 200:
            print("Killswitch OK")
        else:
            print("Killswitch NOT OK")
            exit(1)
    except requests.exceptions.Timeout:
        print("Timeout")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
