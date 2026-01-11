#!/usr/bin/env python3
"""Plot a stacked bar chart showing the number of fruit each person has."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Create a stacked bar graph of fruit counts for three people."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    plt.figure(figsize=(6.4, 4.8))

    people = ["Farrah", "Fred", "Felicia"]
    x = np.arange(len(people))
    width = 0.5

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(x, apples, width=width, color="red", label="apples")
    plt.bar(
        x, bananas, width=width, bottom=apples, color="yellow", label="bananas"
    )
    plt.bar(
        x,
        oranges,
        width=width,
        bottom=apples + bananas,
        color="#ff8000",
        label="oranges",
    )
    plt.bar(
        x,
        peaches,
        width=width,
        bottom=apples + bananas + oranges,
        color="#ffe5b4",
        label="peaches",
    )

    plt.xticks(x, people)
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
