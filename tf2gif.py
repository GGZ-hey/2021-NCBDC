import imageio
import os


def create_gif(img_dir, image_list, gif_name, duration=0.05):
    frames = []
    for image_name in image_list:
        print("image_name={0} img_dir={1}".format(image_name, img_dir))
        frames.append(imageio.imread(img_dir + '/' + image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return



# def main():
img_dir = '/home/Gong.gz/HDD/my_data/2021-NCBDC/figure/heatmap/time_range/24_hour_filled'
duration = 0.5  # 每秒2帧
time_range_list = [0, 7, 9, 11, 13, 15, 17, 19, 21, 24]
time_range_list = list(range(0, 25))
image_list = []
for i in range(len(time_range_list) - 1):
    image_list.append(f'{time_range_list[i]}_{time_range_list[i+1]}.jpg')

gif_name = img_dir + '/heatmap.gif'
create_gif(img_dir, image_list, gif_name, duration)


# if __name__ == '__main__':
#     main()
