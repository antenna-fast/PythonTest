from tqdm import tqdm
import time


if __name__ == '__main__':
    for i in tqdm(range(0, 10), desc='Training: '):
        time.sleep(0.5)
        x = i
        # print(i)
