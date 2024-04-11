import requests


def main():
    # make request to 5.196.23.212/killswitch
    res = requests.get("http://5.196.23.212/killswitch", timeout=5)
    print(res.text)


if __name__ == "__main__":
    main()
