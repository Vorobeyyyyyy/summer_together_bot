import random
import time

import pydirectinput
import pygetwindow as gw
import cv2
import numpy as np
import mss
import statistics
import threading
import dxcam
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

# Название окна игры
window_title = "Лето Вместе"
scale_percent = 100
pydirectinput.PAUSE = 0


def get_window_coordinates(window_title):
    window = gw.getWindowsWithTitle(window_title)
    if window:
        game_window = window[0]
        return game_window.left, game_window.top, game_window.width, game_window.height
    else:
        raise Exception("Окно с заданным заголовком не найдено!")


def capture_screen(left, top, width, height):
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = np.array(sct.grab(monitor))
        return screenshot


def capture_screen_2(camera):
    img_array = camera.get_latest_frame()
    return img_array


def move_capture_zone(left, top, width, height, glitch_small_mode):
    x_offset = 10
    y_offset = 270
    top += y_offset
    left += x_offset
    width -= x_offset * 2
    height -= y_offset + (30 if not glitch_small_mode else 80)
    return left, top, width, height


def resize_image(image):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def load_template(filename):
    template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    template = resize_image(template)
    return template


def detect_contours(gray, good_rects):
    # gray = cv2.GaussianBlur(image, (5, 5), 0)

    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    offset = 0
    for gr in good_rects:
        x_start = max(gr[0] - offset, 0)
        x_end = min(gr[0] + gr[2] + offset, binary.shape[1])
        y_start = max(gr[1] - offset, 0)
        y_end = min(gr[1] + gr[3] + offset, binary.shape[0])

        binary[y_start:y_end, x_start:x_end] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    simplified_contours = []
    for contour in contours:
        epsilon = 1
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if cv2.arcLength(contour, True) > 50:
            simplified_contours.append(approx)

    return simplified_contours


def draw_obstacles(image, contours):
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
    return image


def draw_rects(image, obstacles, color=(0, 255, 0)):
    for (x, y, w, h) in obstacles:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image


def match_template(image, template, th=0.8):
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= th)
    obstacles = [(pt[0], pt[1], w, h) for pt in zip(*loc[::-1])]
    return obstacles


def is_point_inside_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + 90


def remove_contours_inside_template(contours, templates):
    matching_contours = []

    for contour in contours:
        keep_contour = True
        for point in contour:
            for template in templates:
                if is_point_inside_rect(point[0][0], point[0][1], template):
                    keep_contour = False
                    break
            if not keep_contour:
                break
        if keep_contour:
            matching_contours.append(contour)

    return matching_contours


def find_road_center(screen_hsv, road_center):
    # white_pixel_counts = np.sum(binary == 255, axis=0)
    # top_indices = np.argsort(white_pixel_counts)[-8:][::-1]
    # return int(np.mean(top_indices))

    start_col = 0
    if road_center is not None:
        search_width = 20
        # Определяем область интереса вокруг предыдущего центра дороги
        start_col = max(road_center - search_width // 2, 0)
        end_col = min(road_center + search_width // 2, screen_hsv.shape[1])
        screen_hsv = screen_hsv[:, start_col:end_col]

    binary = cv2.inRange(screen_hsv, (15, 20, 245), (30, 30, 248))
    white_pixel_counts = np.sum(binary == 255, axis=0)
    if road_center is not None and white_pixel_counts.max() < 100:
        return None
    top_indices = np.argsort(white_pixel_counts)[-8:][::-1]

    # Возвращаем новый средний центр дороги
    return int(np.mean(top_indices)) + start_col


def group_contours_in_rectangles(contours):
    rectangles = []
    for contour in contours:
        # Вычисление минимального охватывающего прямоугольника
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append([x, y, w, h])
    return rectangles


def filter_obstacles(obstacles, gray):
    filtered_obstacles = []
    for obstacle in obstacles:
        if obstacle[2] > 36 and obstacle[3] > 50:
        # if obstacle[2] > 20 and obstacle[3] > 32:
            filtered_obstacles.append(obstacle)


    return filtered_obstacles


def rectangles_intersect(rect1, rect2):
    """
    Проверяет, пересекаются ли два прямоугольника.
    rect1 и rect2 - кортежи или списки четырех целых чисел: (x, y, width, height).
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False


def remove_good_rects(obstacles, good_rect):
    res = []
    for o in obstacles:
        intersect = False
        if 50 < o[2] < 70:
            for gr in good_rect:
                if rectangles_intersect(o, gr):
                    intersect = True
        if not intersect:
            res.append(o)
    return res


# def get_mask(image):


def get_man_pos(screen_hsv):
    to_find_man = screen_hsv[-40:]
    to_find_man = cv2.inRange(to_find_man, (174, 170, 240), (179, 190, 255))
    indices = np.where(to_find_man == 255)
    x_vals = indices[1]
    return int(statistics.median(x_vals)) if len(x_vals) > 5 else None


def main():
    # mock = False
    mock = True
    glitch_small_mode = False
    if not mock:
        left, top, width, height = get_window_coordinates(window_title)
        left, top, width, height = move_capture_zone(left, top, width, height, glitch_small_mode)
        camera = dxcam.create()
        camera.start(region=(left, top, left + width, top + height), target_fps=165, video_mode=False)

    last_dir = 0
    man_pos = prev_man_pos = None
    move_distances = [0]
    last_time = None
    road_center = None
    image_displayer = ImageDisplayer('Processed')
    image_displayer.start()
    fps = 0
    fail_count = 0

    fps_time = time.time()
    frame_count = 0

    while image_displayer.running and (mock or camera.is_capturing):
        if not mock:
            fail_count += 1
            if fail_count == 100:
                pydirectinput.moveTo(int(left + width / 2), height + 200)
                pydirectinput.click()
                fail_count = -500

        frame_count += 1
        fps_frame_count = 50
        if frame_count % fps_frame_count == 0:
            fps_time_new = time.time()
            fps = 1 / ((fps_time_new - fps_time) / fps_frame_count)
            fps_time = fps_time_new

        now = time.time()
        if last_time is not None:
            frame_time = (now - last_time)
            if frame_time != 0:
                print(f"fps: {fps}. Frame time: {frame_time}")
        last_time = now

        if not mock:
            # screen = capture_screen(left, top, width, height)
            screen = capture_screen_2(camera)
        else:
            screen = cv2.imread('img_7.png')
            screen = screen[270:-72]
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        last_step = time.time()
        print(f"capture time: {last_step - now} s")
        now = last_step

        # screen = resize_image(screen)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        screen_hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)

        last_step = time.time()
        print(f"color transforms: {last_step - now} s")
        now = last_step

        road_center = find_road_center(screen_hsv, road_center)
        if road_center is None:
            continue

        last_step = time.time()
        print(f"road center time: {last_step - now} s")
        now = last_step

        road_width = 225
        star_add_width = 60
        for_starts_width = road_width + star_add_width
        screen = screen[0:screen.shape[0], road_center - road_width:road_center + road_width]
        gray_for_star = gray[0:gray.shape[0], road_center - for_starts_width:road_center + for_starts_width]
        gray = gray[0:gray.shape[0], road_center - road_width:road_center + road_width]
        screen_hsv = screen_hsv[0:screen_hsv.shape[0], road_center - road_width:road_center + road_width]
        screen_height = screen.shape[0]

        if len(gray[0]) == 0:
            continue

        # Ищем положение челика

        man_rect_size = (70, 165) if glitch_small_mode is False else (20, 60)
        if man_pos is not None:
            prev_man_pos = man_pos

        man_pos = get_man_pos(screen_hsv)
        if man_pos is None:
            print('hero not found')
            continue
        if glitch_small_mode:
            man_pos += 3
        if prev_man_pos is None:
            prev_man_pos = man_pos

        if glitch_small_mode:
            man_rect = [man_pos - int(man_rect_size[0] / 2), screen_height - man_rect_size[1], man_rect_size[0],
                        man_rect_size[1]]
        else:
            man_rect = [man_pos + 10 - int(man_rect_size[0] / 2), 935, man_rect_size[0], man_rect_size[1]]

        last_step = time.time()
        print(f"man time: {last_step - now} s")
        now = last_step

        # Ищем звездочки v2
        # cv2.inRange(screen_hsv, (0,0,200), (0,0,p))

        # Ищем звездочки
        _, binary2 = cv2.threshold(gray_for_star, 100, 255, cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont2 = []
        for contour in contours2:
            approx = cv2.approxPolyDP(contour, 3, True)
            if 20 <= approx.shape[0] <= 22 and cv2.arcLength(approx, True) < 320:
                cont2.append(approx)
        good_rect = group_contours_in_rectangles(cont2)
        good_rect_for_remove = []
        for rect in good_rect:
            rect[0] -= star_add_width
            rect_offset = 0 if glitch_small_mode else 10


            rect[0] -= int(rect_offset / 2)
            rect[1] -= int(rect_offset / 2)
            rect[2] += rect_offset
            rect[3] += rect_offset

            rect_offset = 15 if screen_height - rect[1] < 150 else 0
            rect_offset += 10 if glitch_small_mode else 0

            good_rect_for_remove.append([
                rect[0] - int(rect_offset / 2),
                rect[1] - int(rect_offset / 2),
                rect[2] + rect_offset,
                rect[3] + rect_offset
            ])

        last_step = time.time()
        print(f"star time: {last_step - now} s")
        now = last_step

        contours = detect_contours(gray, good_rect_for_remove + [man_rect])
        last_step = time.time()
        print(f"contours time: {last_step - now} s")
        now = last_step
        obstacles = filter_obstacles(group_contours_in_rectangles(contours), screen)
        # obstacles = remove_good_rects(obstacles, good_rect)
        last_step = time.time()
        print(f"obstacles time: {last_step - now} s")
        now = last_step

        # draw_rects(screen, res)
        draw_rects(screen, [man_rect])
        draw_rects(screen, good_rect, (255, 255, 0))
        draw_rects(screen, good_rect_for_remove, (255, 255, 0))
        draw_rects(screen, obstacles, (255, 0, 0))

        line_step = 5
        line_count = 92
        line_start = 650
        center_line = int(line_count / 2)
        lines = []
        to_go_width = 19
        min_to_go_width = 3
        star_border_dead_zone = 15

        for i in range(line_count):
            line_x = i * line_step
            bad = hit_any_rect(line_start, line_x, obstacles)
            with_points = hit_any_rect(line_start, line_x, good_rect)
            lines.append(MoveLine(line_x, not bad, with_points))

        count = 0
        for i in range(len(lines)):
            if lines[i].good:
                count += 1
            else:
                count = 0
            if count >= to_go_width:
                lines[i - int(to_go_width / 2)].to_move = True

        count = 0
        for i, line in enumerate(lines):
            if line.to_move:
                count += 1
            else:
                if count < min_to_go_width:
                    for j in range(i - min_to_go_width + 1, i + 1):
                        lines[j].to_move = False
                count = 0

        nearest_line = None
        li = None

        move_distance = abs(prev_man_pos - man_pos)
        print(f"move_distance: {move_distance}")
        if move_distance > 2 and last_dir in [-1, 1]:
            move_distances.append(move_distance)
        if len(move_distances) > 50:
            move_distances.pop(0)
        # linear_advance = int(statistics.median(move_distances))
        linear_advance = 15

        adv_man_pos = linear_advance * last_dir + man_pos

        for i, line in enumerate(lines):
            x = line.pos
            if line.to_move and (nearest_line is None or abs(nearest_line.pos - adv_man_pos) > abs(x - adv_man_pos)):
                nearest_line = line
                li = i
            cv2.line(screen, (x, line_start), (x, 1100), line.color(), 1)

        if nearest_line is not None:

            # Поиск линии с бонусом
            nearest_bonus_l = None
            for i in range(li, star_border_dead_zone - 1, -1):
                if lines[i].to_move:
                    if lines[i].with_points:
                        nearest_bonus_l = i
                        break
                else:
                    break
            nearest_bonus_r = None
            for i in range(li, len(lines) - star_border_dead_zone):
                if lines[i].to_move:
                    if lines[i].with_points:
                        nearest_bonus_r = i
                        break
                else:
                    break
            nearest_bonus = None
            if nearest_bonus_l is None and nearest_bonus_r is not None:
                nearest_bonus = nearest_bonus_r
            elif nearest_bonus_r is None and nearest_bonus_l is not None:
                nearest_bonus = nearest_bonus_l
            elif nearest_bonus_r is not None and nearest_bonus_l is not None:
                if nearest_bonus_r - li < li - nearest_bonus_l:
                    nearest_bonus = nearest_bonus_r
                else:
                    nearest_bonus = nearest_bonus_l
            if nearest_bonus is not None and nearest_bonus != center_line:
                for i in range(nearest_bonus, center_line, 1 if nearest_bonus < center_line else -1):
                    if lines[i].with_points and lines[i].to_move:
                        nearest_bonus = i
            if nearest_bonus is not None:
                nearest_line = lines[nearest_bonus]

            # Смещение к самой левой линии
            if nearest_bonus is None:
                li_tmp = li
                for i in range(1, abs(int(line_count / 2) - li_tmp) + 1):
                    pot_li = li_tmp + i if li_tmp < int(line_count / 2) else li_tmp - i
                    if lines[pot_li].to_move:
                        nearest_line = lines[pot_li]

            cv2.line(screen, (nearest_line.pos, line_start), (nearest_line.pos, 1100), (255, 255, 0), 2)
            cv2.line(screen, (adv_man_pos, 1000), (adv_man_pos, 1100), (0, 0, 255), 4)

            # print(nearest_line[0], man_pos, adv_man_pos)
            approx_line = lines[round(adv_man_pos / line_step)]
            current_line = lines[round(man_pos / line_step)]
            if ((abs(nearest_line.pos - adv_man_pos) < line_step * 4 and approx_line.to_move)
                    or (current_line.to_move and not approx_line.to_move)):
                cur_dir = 0
            elif nearest_line.pos < adv_man_pos:
                cur_dir = -1
            else:
                cur_dir = 1

            if cur_dir != last_dir:
                print(f"Change dir to {cur_dir}")
                # thread = threading.Thread(target=change_dir, args=(cur_dir, last_dir)).start()
                change_dir(cur_dir, last_dir)
            last_dir = cur_dir

        last_step = time.time()
        print(f"lines time: {last_step - now} s")
        now = last_step

        write([
            f"{int(fps)} FPS",
            f"Linear advance: {linear_advance}",
            f"Man pos: {man_pos}",
            f"Cur dir: {last_dir}",
        ], screen)
        image_displayer.update_image(screen)
        fail_count = 0

    if not mock:
        camera.stop()


def write(text, screen):
    for i, line in enumerate(text):
        cv2.putText(screen, line, (20, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, 1)


def hit_any_rect(line_start, line_x, obstacles):
    hit = False
    for o in obstacles:
        if o[0] < line_x < o[0] + o[2] and o[1] + o[3] > line_start:
            hit = True
    return hit


sem = threading.Semaphore(value=1)
show_sem = threading.Semaphore(value=1)


class MoveLine:
    def __init__(self, pos, good, with_points):
        self.pos = pos
        self.good = good
        self.to_move = False
        self.with_points = with_points

    def color(self):
        blue = 255 if self.with_points else 0
        green = 255 if self.good else 0
        red = 255 if not self.to_move else 0
        return blue, green, red


class ImageDisplayer(threading.Thread):
    def __init__(self, window_name):
        threading.Thread.__init__(self)
        self.window_name = window_name
        self.img = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        cv2.namedWindow(self.window_name)
        while self.running:
            self.lock.acquire()
            if self.img is not None:
                cv2.imshow(self.window_name, self.img)
            self.lock.release()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        cv2.destroyAllWindows()

    def update_image(self, img):
        self.lock.acquire()
        self.img = img
        self.lock.release()

    def stop(self):
        self.running = False
        self.join()


def change_dir(cur_dir, last_dir):
    sem.acquire()
    if last_dir == -1:
        pydirectinput.keyUp('a')
        print("left released")
    if last_dir == 1:
        pydirectinput.keyUp('d')
        print("right released")
    if cur_dir == -1:
        pydirectinput.keyDown('a')
        print("left pressed")
    if cur_dir == 1:
        pydirectinput.keyDown('d')
        print("right pressed")
    sem.release()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
