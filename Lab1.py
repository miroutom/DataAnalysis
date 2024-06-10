from typing import Dict

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset() -> DataFrame:
    """
    This function loads dataset from 'diamonds.csv' file
    :return: Completed dataset
    """
    return pd.read_csv('diamonds.csv')


def clean_dataset(diamonds: DataFrame) -> DataFrame:
    """
    This function cleans all data in dataset: delete missing values, duplicates and outliers
    if needed
    :param diamonds: dataset
    :return: None
    """

    diamonds.dropna(inplace=True)

    del diamonds['Unnamed: 0']
    diamonds.drop_duplicates(inplace=True)

    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

    Q1 = diamonds[numeric_columns].quantile(0.25)
    Q3 = diamonds[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    return diamonds[~((diamonds[numeric_columns] < (Q1 - 1.5 * IQR)) |
                      (diamonds[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]


def define_central_measures_for_numeric(diamonds: DataFrame) -> DataFrame:
    """
    This function counts mean, median and mode for all numeric columns
    :param diamonds: dataset
    :return: Dictionary key-value, where key = col, value = dict(mean, median, mode)
    """

    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    central_measures = pd.DataFrame(index=("Mean", "Median", "Mode"))

    for col in numeric_columns:
        mean = diamonds[col].mean()
        median = diamonds[col].median()
        mode = diamonds[col].mode()[0]

        central_measures[col] = [mean, median, mode]

    # -------------------------------------------------------------------------------------
    # Analysis
    # 1. carat. Среднее значение незначительно больше медианы, а медиана значительно больше
    # моды, что указывает на правостороннюю ассиметрию (несколько крупных бриллиантов,
    # который поднимают среднее значение)

    # 2. depth. Все три меры очень близки друг к другу по значению, что указывает на
    # симметричное распределение данных

    # 3. table. Среднее значение и медиана очень близки друг к другу, а мода незначительно
    # меньше, что указывает на почти симметричное распределение с небольшой правосторонней
    # ассиметрией

    # 4. price. Среднее значение сильно превышает медиану, что указывает на правостороннюю
    # ассиметрию. Мода также значительно ниже среднего значения и медианы, это говорит о
    # наличии большого количества недорогих алмазов. Кроме того, по предыдущим показателям
    # известно, что количество мелких алмазов превалирует

    # 5. x. Среднее значение и медиана имеют незначительную разницу, что указывает на
    # симметричное распределение. Мода значительно ниже, это говорит о наличии большого
    # количества более коротких алмазов.

    # 6. y. Среднее значение и медиана имеют незначительную разницу, что указывает на
    # симметричное распределение. Мода значительно ниже, это говорит о наличии большого
    # количества более узких алмазов.

    # 7. z. Среднее значение и медиана имеют незначительную разницу, что указывает на
    # симметричное распределение. Мода значительно ниже, это говорит о наличии большого
    # количества более неглубоких алмазов.

    # Общий вывод: для переменных carat, price  наблюдается правостороняя ассиметрия,
    # что указывает на наличие нескольких значений, сильно превыщающих большинство других
    # (например, крупные или дорогие алмазы).
    # Для остальных переменных характерно более симметричное распределение.
    # -------------------------------------------------------------------------------------

    return central_measures


def visualize_data_with_histograms(diamonds: DataFrame) -> None:
    """
    This function creates histograms for carat, depth and price columns
    :param diamonds: dataset
    :return: None
    """
    plt.figure(figsize=(12, 7))
    plt.hist(diamonds['carat'], bins=50, color='green', edgecolor='black')
    plt.title('Carat histogram')
    plt.xlabel('Weight [carat]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.hist(diamonds['depth'], bins=50, color='blue', edgecolor='black')
    plt.title('Depth histogram')
    plt.xlabel('Depth [%]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.hist(diamonds['price'], bins=50, color='yellow', edgecolor='black')
    plt.title('Histogram for price')
    plt.xlabel('Price [USD]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def visualize_data_with_boxplot(diamonds: DataFrame) -> None:
    """
    This functions creates box plots with diamonds price depends on cut and diamond price depends on color
    :param diamonds: dataset
    :return: None
    """
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='cut', y='price', data=diamonds, palette='inferno', hue='cut')
    plt.title('Price depends on cut')
    plt.xlabel('Cut')
    plt.ylabel('Price [USD]')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='color', y='price', data=diamonds, palette='magma', hue='color')
    plt.title('Price depends on color')
    plt.xlabel('Color')
    plt.ylabel('Price [USD]')
    plt.grid(True)
    plt.show()


def visualize_data_with_scatterplot(diamonds: DataFrame) -> None:
    """
    This functions creates scatter plot with diamond's weight and price relationship
    :param diamonds: dataset
    :return: None
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(diamonds['carat'], diamonds['price'], alpha=0.5, color='purple')
    plt.title('Weight and price relationship')
    plt.xlabel('Weight [carat]')
    plt.ylabel('Price [USD]')
    plt.grid(True)
    plt.show()


def define_variability_measures(diamonds: DataFrame) -> DataFrame:
    """
    This function counts variability measures (standard deviation and range) for carat, depth and price columns
    :param diamonds: dataset
    :return: Dictionary key-value, where key=col, dict = dict(std_deviation, range)
    """
    columns = ['carat', 'depth', 'price']
    variability_measures = pd.DataFrame(index=("STD", "Range"))

    for col in columns:
        standard_deviation = diamonds[col].std()
        dataset_range = diamonds[col].max() - diamonds[col].min()

        variability_measures[col] = [standard_deviation, dataset_range]

    # -------------------------------------------------------------------------------------
    # Analysis
    # 1. STD. У переменных carat и depth стандартное отклонения является небольшим.
    # Это указывает на то, что размеры алмазов в датасете находятся в основном
    # к близкому значению. Глубина алмазов также не сильно варьируется. У переменной price
    # наблюдается значительное отклонение, что указывает на сильный разброс
    # цен на бриллианты.

    # 2. Range. Для carat: диапазон значений составляет 1.8, что указывает на
    # значительное различие в размерах алмазов в датасете.
    # Для depth: Диапазон значений
    # глубины составляет 5.9. Это относительно небольшой диапазон, указывающий на то,
    # что глубина алмазов в среднем находится в пределах этого значения.
    # Для price: Диапазон цен указывает на большую вариативность. Это может зависеть от
    # различных характеристик алмазов. Также можно сделать вывод, что в наличие присутствуют,
    # как очень дорогие, так и сравнительно дешевые алмазы.
    # -------------------------------------------------------------------------------------

    return variability_measures


def visualize_correlation_matrix(diamonds: DataFrame) -> None:
    """
    This function creates correlation matrix for numeric columns and visualize it using heat map
    :param diamonds: dataset
    :return: None
    """
    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

    correlation_matrix = diamonds[numeric_columns].corr()
    print(correlation_matrix)

    plt.figure(figsize=(12, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='plasma', fmt=".2f", linewidths=.5)
    plt.title('Correlation matrix')
    plt.show()

    # -------------------------------------------------------------------------------------
    # Analysis
    # Высокая положительная корреляция между carat и price указывает на то, что более
    # крупные алмазы стоят дороже. Это ожидаемое поведение, ведь весь является
    # одним из ключевых определителей цены.

    # Высокая положительная корреляция между carat и x, y, z. Данные корреляции показывают,
    # что вес алмаза сильно связан с его измерениями. Это ожидаемое поведение, ведь чем
    # больше алмаз, тем крупнее его вес.

    # Высокая положительная корреляция между price и x, y, z. Данные корреляции показывают,
    # что цена алмаза сильно связана с его измерениями. Это ожидаемое поведение, ведь чем
    # больше алмаз, тем крупнее его вес, а следовательно и цена.

    # Негативная корреляция между depth и остальными переменными. Данные корреляции показывают,
    # что глубина алмаза имеет практические отсутствующие или очень слабые корреляции с
    # другими пOutlеременными. Это указывает на то, что характеристики алмаза не сильно связаны
    # с его глубиной.
    # -------------------------------------------------------------------------------------


def main() -> None:
    """
    This is the main function
    :return: None
    """
    diamonds = clean_dataset(load_dataset())

    print(f"Central measures for numeric columns\n{define_central_measures_for_numeric(diamonds)}")
    print("\n")
    print(f"Variability measures\n{Outldefine_variability_measures(diamonds)}")
    print("\n")

    visualize_data_with_histograms(diamonds)
    visualize_data_with_boxplot(diamonds)
    visualize_data_with_scatterplot(diamonds)
    visualize_correlation_matrix(diamonds)


if __name__ == '__main__':
    main()
