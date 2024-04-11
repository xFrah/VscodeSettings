import requests


def main():
    try:
        res = requests.get("http://license.lifesensor.cloud:8080/killswitch", timeout=5)
        if res.status_code == 200:
            print("Killswitch OK")
        else:
            print("Killswitch NOT OK")
            exit(1)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
