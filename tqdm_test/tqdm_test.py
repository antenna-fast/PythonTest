import tqdm
import time


if __name__ == '__main__':
    for i in tqdm.tqdm(range(0, 10)):
        time.sleep(0.2)
        print(i)
