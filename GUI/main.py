#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pyautogui
import copy

from hearts.game_engine import *
from poker.poker import *

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.core.window import Window

from PIL import ImageGrab
from ultralytics import YOLO
from typing import List, Tuple, Union



IMG_SIZE = 640
CARD_SUITS = ["spades", "hearts", "diamonds", "clubs"]
CARD_SUITS_SHORT = ["S", "H", "D", "C"]
CARD_SYMBOLS = [str(n) for n in range(2, 14 + 1)]
CLASSES = [f"{value}_of_{suit}" for suit in CARD_SUITS for value in CARD_SYMBOLS]


Builder.load_file("layout.kv")


class Card2:
    def __init__(self, coords, name, name_short):
        self.coords = coords
        self.name = name
        self.name_short = name_short


class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super(MenuScreen, self).__init__(**kwargs)
        self.model_symbols = YOLO("weights_symbols.pt")
        self.model_suits = YOLO("weights_suits.pt")
        self.cards = []
        self.file_name = "test.jpg"
        self.img = None
        self.a = None
        self.b = None
        self.mouse_coords = None
        self.snip_coords = []
        self.cards1 = []
        self.cards2 = []

    def take_screenshot(self) -> None:
        self.file_name = "screenshot.png"
        bbox = self.snip_coords if len(self.snip_coords) == 4 else None
        screenshot = ImageGrab.grab(bbox=bbox)
        screenshot.save(self.file_name)
        screenshot.close()
        table_img = self.ids["table_img"]
        table_img.source = self.file_name
        table_img.reload()

    @staticmethod
    def get_classes_polygons(model, file_name_part, classes, polygons, x1, y1):
        boxes = model.predict(source=file_name_part, conf=0.001, imgsz=IMG_SIZE)[0].boxes
        classes.extend(list(boxes.cls))

        for polygon, conf in zip(list(boxes.xywh), list(boxes.conf)):
            x, y, w, h = polygon.cpu().numpy()
            polygons.append(((x + x1, y + y1, w, h), conf))

        return classes, polygons

    @staticmethod
    def get_polygons(polygons_raw):
        polygons = []

        for m in range(len(polygons_raw)):
            is_unique = True

            for n in range(m):
                x1, y1, _, _ = polygons_raw[m][0]
                x2, y2, w, h = polygons_raw[n][0]

                if x2 - w / 2 < x1 < x2 + w / 2 and y2 - h / 2 < y1 < y2 + h / 2:
                    is_unique = False
                    break

            if is_unique:
                polygons.append(polygons_raw[m])

        return polygons

    def put_text(self, text, x, y, w, h):
        coords = (int(x + w / 2), int(y + h / 2))
        color = (255, 0, 0)
        self.img = cv2.putText(self.img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, self.a / 500, color, 4, cv2.LINE_AA)

    def display_img(self, ax):
        ax.imshow(self.img)
        ax.axis('off')
        plt.savefig("image.jpg", bbox_inches='tight', pad_inches=0)
        table_img = self.ids["table_img"]
        table_img.source = "image.jpg"
        table_img.reload()

        cards = []

        for card in self.cards:
            new = True

            for i in range(len(cards)):
                if cards[i].name == card.name:
                    x1, y1, w1, h1 = card.coords
                    x2, y2, w2, h2 = cards[i].coords
                    new = False

                    if y1 > y2:
                        cards.pop(i)
                        cards.append(card)

            if new:
                cards.append(card)

        with open('..\hearts\input_cards.txt', 'w') as f1:
            with open('..\hearts\played_cards.txt', 'w') as f2:
                cards1 = []
                cards2 = []
                H = self.img.shape[0]

                for card in cards:
                    suit = card.name_short[0]
                    symbol = card.name_short[1:len(card.name_short)]
                    name = f"{symbol},{suit}"

                    x, y, w, h = card.coords

                    if y < H - 100:
                        cards2.append(name)

                    else:
                        cards1.append(name)

                f1.write("\n".join(cards1[:13]))
                f2.write("\n".join(cards2[:3]))
                self.cards1 = cards1[:2]
                self.cards2 = cards2[:5]

    def predict(self) -> None:
        self.cards = []
        self.img = cv2.cvtColor(cv2.imread(self.file_name), cv2.COLOR_BGR2RGB)
        H, W = self.img.shape[:2]
        self.a, self.b = max(min(H, W), IMG_SIZE), max(max(H, W), IMG_SIZE)
        y1, y2, x1, x2 = 0, min(self.a, H), 0, min(self.a, W)
        N = int(self.b / self.a)

        polygons_suits_raw = []
        classes_suits = []
        polygons_symbols_raw = []
        classes_symbols = []

        for n in range(1, N + 1):
            file_name, extension = self.file_name.split('.')
            file_name_part = f"{file_name}_{n}.{extension}"
            img_part = np.ones((self.a, self.b, 3)) * 255
            img_part[0:y2 - y1, 0:x2 - x1] = self.img[y1:y2, x1:x2]
            cv2.imwrite(file_name_part, img_part)
            classes_suits, polygons_suits_raw = self.get_classes_polygons(self.model_suits, file_name_part,
                                                                          classes_suits, polygons_suits_raw, x1, y1)
            classes_symbols, polygons_symbols_raw = self.get_classes_polygons(self.model_symbols, file_name_part,
                                                                              classes_symbols, polygons_symbols_raw, x1, y1)
            y1 = y1 + self.a if y2 < H else y1
            y2 = min(y2 + self.a, H) if y2 < H else y2
            x1 = x1 + self.a if x2 < W else x1
            x2 = min(x2 + self.a, W) if x2 < W else x2

        polygons_suits = self.get_polygons(polygons_suits_raw)
        polygons_symbols = self.get_polygons(polygons_symbols_raw)
        fig, ax = plt.subplots(figsize=(5, 5))

        for m in range(len(polygons_suits)):
            for n in range(len(polygons_symbols)):
                (x1, y1, w1, h1), conf1 = polygons_suits[m]
                (x2, y2, w2, h2), conf2 = polygons_symbols[n]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if dist < max(w1, h1, w2, h2):
                    x = min(x1 - w1 / 2, x2 - w2 / 2)
                    y = min(y1 - h1 / 2, y2 - h2 / 2)
                    w = max(x1 + w1 / 2, x2 + w2 / 2) - x
                    h = max(y1 + h1 / 2, y2 + h2 / 2) - y
                    suit_idx = int(classes_suits[m])
                    symbol_idx = int(classes_symbols[n])
                    card_name = f"{CARD_SYMBOLS[symbol_idx]}_of_{CARD_SUITS[suit_idx]}"
                    card_name_short = f"{CARD_SUITS_SHORT[suit_idx]}{CARD_SYMBOLS[symbol_idx]}"
                    self.put_text(card_name_short, x, y, w, h)
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    card = Card2((x, y, w, h), card_name, card_name_short)
                    self.cards.append(card)

        self.display_img(ax)

    def change(self) -> None:
        if self.img is None:
            return None

        box = self.ids["cards_layout"]
        box.clear_widgets()

        x0, y0 = self.mouse_coords
        y0 = self.ids["main_layout"].height - y0
        x0 = x0 / self.ids["img_layout"].width * self.img.shape[1]
        y0 = y0 / self.ids["img_layout"].height * self.img.shape[0]

        for card in self.cards:
            x, y, w, h = card.coords

            if x <= x0 <= x + w and y <= y0 <= y0 + h:
                img = Image(allow_stretch=True, keep_ratio=False, source=f"cards/{card.name}.png")
                box.add_widget(img)
                box.add_widget(Button(text="Change suit",
                                      on_press=lambda _, img_=img, card_=card: self.change_suit(img_, card_)))
                box.add_widget(Button(text="Change symbol",
                                      on_press=lambda _, img_=img, card_=card: self.change_symbol(img_, card_)))
                break

    def change_suit(self, img, card) -> None:
        symbol, old_suit = card.name.split("_of_")
        suit_mapping = {"spades": "hearts", "hearts": "diamonds", "diamonds": "clubs", "clubs": "spades"}
        suit = suit_mapping[old_suit]
        self.change_text(img, card, symbol, suit)

    def change_symbol(self, img, card) -> None:
        old_symbol, suit = card.name.split("_of_")
        symbol_mapping = {"jack": "11", "queen": "12", "king": "13", "ace": "14"}
        symbol = int(symbol_mapping[old_symbol]) + 1 if old_symbol in symbol_mapping else int(old_symbol) + 1
        symbol = "2" if symbol > 14 else str(symbol)
        self.change_text(img, card, symbol, suit)

    def change_text(self, img, ref_card, symbol, suit):
        img.source = f"cards/{symbol}_of_{suit}.png"
        img.reload()
        self.img = cv2.cvtColor(cv2.imread(self.file_name), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(5, 5))
        cards = []

        for card in self.cards:
            x, y, w, h = card.coords

            if card == ref_card:
                card.name = f"{symbol}_of_{suit}"
                card.name_short = f"{suit[0].upper()}{symbol}"

            self.put_text(card.name_short, x, y, w, h)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            cards.append(card)

        self.cards = cards
        self.display_img(ax)

    def run(self) -> None:
        game_type = self.ids["game_type"].text
        print(game_type)

        if game_type == "Hearts":
            engine_ = HeartsEngine.create()
            best_card = engine_.play(50)
            self.ids["result_label"].text = str(best_card)

        else:
            print(self.cards1, self.cards2)
            # if self.ids["n_players"].text != "":
                # res = count_probability(self.cards1, self.cards2, int(self.ids["n_players"].text))
            res = count_probability(['8,H', '6,H'], ['6,S', '7,S', '9,S', '5,S', '10,C'], 5)
            # res = count_probability(["2,C", "5,S"], ["3,S", "5,H", "13,H"], 5)
            print(res)
            self.ids["result_label"].text = f"{res[0]},{res[1]}"

    def change_game(self) -> None:
        game_type = self.ids["game_type"].text
        self.ids["game_type"].text = "Poker" if game_type == "Hearts" else "Hearts"


class TestApp(App):
    def __init__(self, **kwargs):
        super(TestApp, self).__init__(**kwargs)
        self.screen = None
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self.root)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, *args):
        key = args[2]

        if key == "v":
            self.screen.take_screenshot()

        elif key == "b":
            self.screen.predict()

        elif key == "z":
            self.screen.run()

        elif key == "n":
            self.screen.change()

        elif key == "w":
            if len(self.screen.snip_coords) == 4:
                x1, y1, x2, y2 = self.screen.snip_coords
                self.screen.snip_coords = [x2, y2]

            pos = pyautogui.position()
            x, y = pos.x, pos.y
            self.screen.snip_coords.extend([x, y])

    def build(self):
        sm = ScreenManager()
        self.screen = MenuScreen(name="menu")
        sm.add_widget(self.screen)
        Window.bind(mouse_pos=self.assign)

        return sm

    def assign(self, w, p) -> None:
        self.screen.mouse_coords = p


if __name__ == "__main__":
    TestApp().run()
