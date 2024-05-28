#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.core.window import Window

from PIL import ImageGrab
from ultralytics import YOLO
from typing import List, Tuple, Union


CARD_SUITS = ["spades", "hearts", "diamonds", "clubs"]
CARD_VALUES = [n for n in range(2, 14 + 1)]
CLASSES = [f"{value}_of_{suit}" for suit in CARD_SUITS for value in CARD_VALUES]


Builder.load_file("layout.kv")


class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super(MenuScreen, self).__init__(**kwargs)
        self.model = YOLO("weights.pt")
        self.setup_ui()
        self.cards = []

    def setup_ui(self) -> None:
        pass

    def take_screenshot(self) -> None:
        screenshot = ImageGrab.grab()
        screenshot.save("screenshot.png")
        screenshot.close()
        table_img = self.ids["table_img"]
        table_img.source = "screenshot.png"
        table_img.reload()

    def predict(self) -> None:
        results = self.model.predict(source="screenshot.png", conf=0.25, imgsz=720)

        img = cv2.imread("screenshot.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_facecolor((0, 0, 0))
        ax.imshow(img)

        for polygon in results[0].boxes.xyxy:
            x1, y1, x2, y2 = polygon.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        for idx in results[0].boxes.cls:
            cls = CLASSES[int(idx)]

            if cls not in self.cards:
                self.cards.append(cls)

        plt.show()

        box = self.ids["cards_layout"]
        box.clear_widgets()

        for card in self.cards:
            box.rows += 1
            img = Image(allow_stretch=True, keep_ratio=False, source=f"cards/{card}.png")
            box.add_widget(img)
            box.add_widget(Button(text="Change suit", on_press=lambda _, img_=img: self.change_suit(img_)))
            box.add_widget(Button(text="Change symbol", on_press=lambda _, img_=img: self.change_symbol(img_)))

    def change_suit(self, img) -> None:
        symbol, suit = img.source.split("/")[1].split(".")[0].split("_of_")
        suit_mapping = {"spades": "hearts", "hearts": "diamonds", "diamonds": "clubs", "clubs": "spades"}
        suit = suit_mapping[suit]
        img.source = f"cards/{symbol}_of_{suit}.png"

    def change_symbol(self, img) -> None:
        symbol, suit = img.source.split("/")[1].split(".")[0].split("_of_")
        symbol_mapping = {"jack": "11", "queen": "12", "king": "13", "ace": "14"}

        if symbol in symbol_mapping:
            symbol = symbol_mapping[symbol]

        symbol = int(symbol) + 1
        symbol = "2" if symbol > 14 else str(symbol)
        img.source = f"cards/{symbol}_of_{suit}.png"

    def run(self) -> None:
        pass


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

    def build(self):
        sm = ScreenManager()
        self.screen = MenuScreen(name="menu")
        sm.add_widget(self.screen)

        return sm


if __name__ == "__main__":
    TestApp().run()
