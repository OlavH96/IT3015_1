import csv
import Config


def convertArrayTypes(array):
    return list(map(lambda e: Config.parseArgType(e), array))


def read(path, delimiter=','):
    file = open(path)

    csvFile = csv.reader(file, delimiter=delimiter)

    return [convertArrayTypes(row) for row in csvFile]
