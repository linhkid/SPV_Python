import random
import math
import numpy as np
from scipy.special import comb
from sklearn.utils import shuffle
from sklearn import datasets
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple

class SUtils:
    minNumThreads = 4000
    displayPerfAfterInstances = 1000
    perfOutput = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+*`~!@#$%^&_|:;'?"
    m_Limit = 1
    seed = 1

    @staticmethod
    def computeMutualInformation(u1: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        m = 0
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            xcount = 0
            for y in range(nc):
                index = dParameters_.getAttributeIndex(u1, u1val, y)
                xcount += dParameters_.getCountAtFullIndex(index)
            for c in range(nc):
                avyCount = dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, c))
                ycount = dParameters_.getCountAtFullIndex(c)
                if avyCount > 0:
                    m += (avyCount / N) * math.log( avyCount / ( xcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformationPerFeatureValue(u1: int, u1valin: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        xcount = 0
        xNOTcount = 0
        avyCount = [0] * nc
        avyNOTcount = [0] * nc
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            if u1val == u1valin:
                for y in range(nc):
                    index = dParameters_.getAttributeIndex(u1, u1val, y)
                    xcount += dParameters_.getCountAtFullIndex(index)
                    avyCount[y] += dParameters_.getCountAtFullIndex(index)
            else:
                for y in range(nc):
                    index = dParameters_.getAttributeIndex(u1, u1val, y)
                    xNOTcount += dParameters_.getCountAtFullIndex(index)
                    avyNOTcount[y] += dParameters_.getCountAtFullIndex(index)
        m = 0
        for c in range(nc):
            ycount = dParameters_.getCountAtFullIndex(c)
            if avyCount[c] > 0:
                m += (avyCount[c] / N) * math.log( avyCount[c] / ( xcount/N * ycount ) ) / math.log(2)
            if avyNOTcount[c] > 0:
                m += (avyNOTcount[c] / N) * math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformation(u1: int, u2: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        m = 0
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                xcount = 0
                for y in range(nc):
                    index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y)
                    xcount += dParameters_.getCountAtFullIndex(index)
                for c in range(nc):
                    avyCount = dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, c))
                    ycount = dParameters_.getCountAtFullIndex(c)
                    if avyCount > 0:
                        m += (avyCount / N) * math.log( avyCount / ( xcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformationPerFeatureValue(u1: int, u1valin: int, u2: int, u2valin: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        xcount = 0
        xNOTcount = 0
        avyCount = [0] * nc
        avyNOTcount = [0] * nc
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                if u1val == u1valin and u2val == u2valin:
                    for y in range(nc):
                        index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y)
                        xcount += dParameters_.getCountAtFullIndex(index)
                        avyCount[y] += dParameters_.getCountAtFullIndex(index)
                else:
                    for y in range(nc):
                        index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y)
                        xNOTcount += dParameters_.getCountAtFullIndex(index)
                        avyNOTcount[y] += dParameters_.getCountAtFullIndex(index)
        m = 0
        for c in range(nc):
            ycount = dParameters_.getCountAtFullIndex(c)
            if avyCount[c] > 0:
                m += (avyCount[c] / N) * math.log( avyCount[c] / ( xcount/N * ycount ) ) / math.log(2)
            if avyNOTcount[c] > 0:
                m += (avyNOTcount[c] / N) * math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformation(u1: int, u2: int, u3: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        m = 0
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    xcount = 0
                    for y in range(nc):
                        index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y)
                        xcount += dParameters_.getCountAtFullIndex(index)
                    for c in range(nc):
                        avyCount = dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c))
                        ycount = dParameters_.getCountAtFullIndex(c)
                        if avyCount > 0:
                            m += (avyCount / N) * math.log( avyCount / ( xcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformationPerFeatureValue(u1: int, u1valin: int, u2: int, u2valin: int, u3: int, u3valin: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        xcount = 0
        xNOTcount = 0
        avyCount = [0] * nc
        avyNOTcount = [0] * nc
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    if u1val == u1valin and u2val == u2valin and u3val == u3valin:
                        for y in range(nc):
                            index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y)
                            xcount += dParameters_.getCountAtFullIndex(index)
                            avyCount[y] += dParameters_.getCountAtFullIndex(index)
                    else:
                        for y in range(nc):
                            index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y)
                            xNOTcount += dParameters_.getCountAtFullIndex(index)
                            avyNOTcount[y] += dParameters_.getCountAtFullIndex(index)
        m = 0
        for c in range(nc):
            ycount = dParameters_.getCountAtFullIndex(c)
            if avyCount[c] > 0:
                m += (avyCount[c] / N) * math.log( avyCount[c] / ( xcount/N * ycount ) ) / math.log(2)
            if avyNOTcount[c] > 0:
                m += (avyNOTcount[c] / N) * math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformation(u1: int, u2: int, u3: int, u4: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        m = 0
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    for u4val in range(dParameters_.getParamsPerAtt()[u4]):
                        xcount = 0
                        for y in range(nc):
                            index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y)
                            xcount += dParameters_.getCountAtFullIndex(index)
                        for c in range(nc):
                            avyCount = dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c))
                            ycount = dParameters_.getCountAtFullIndex(c)
                            if avyCount > 0:
                                m += (avyCount / N) * math.log( avyCount / ( xcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformationPerFeatureValue(u1: int, u1valin: int, u2: int, u2valin: int, u3: int, u3valin: int, u4: int, u4valin: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        xcount = 0
        xNOTcount = 0
        avyCount = [0] * nc
        avyNOTcount = [0] * nc
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    for u4val in range(dParameters_.getParamsPerAtt()[u4]):
                        if u1val == u1valin and u2val == u2valin and u3val == u3valin and u4val == u4valin:
                            for y in range(nc):
                                index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y)
                                xcount += dParameters_.getCountAtFullIndex(index)
                                avyCount[y] += dParameters_.getCountAtFullIndex(index)
                        else:
                            for y in range(nc):
                                index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y)
                                xNOTcount += dParameters_.getCountAtFullIndex(index)
                                avyNOTcount[y] += dParameters_.getCountAtFullIndex(index)
        m = 0
        for c in range(nc):
            ycount = dParameters_.getCountAtFullIndex(c)
            if avyCount[c] > 0:
                m += (avyCount[c] / N) * math.log( avyCount[c] / ( xcount/N * ycount ) ) / math.log(2)
            if avyNOTcount[c] > 0:
                m += (avyNOTcount[c] / N) * math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformation(u1: int, u2: int, u3: int, u4: int, u5: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        m = 0
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    for u4val in range(dParameters_.getParamsPerAtt()[u4]):
                        for u5val in range(dParameters_.getParamsPerAtt()[u5]):
                            xcount = 0
                            for y in range(nc):
                                index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y)
                                xcount += dParameters_.getCountAtFullIndex(index)
                            for c in range(nc):
                                avyCount = dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c))
                                ycount = dParameters_.getCountAtFullIndex(c)
                                if avyCount > 0:
                                    m += (avyCount / N) * math.log( avyCount / ( xcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def computeMutualInformationPerFeatureValue(u1: int, u1valin: int, u2: int, u2valin: int, u3: int, u3valin: int, u4: int, u4valin: int, u5: int, u5valin: int, dParameters_: Parameter) -> float:
        nc = dParameters_.getNC()
        N = dParameters_.getN()
        xcount = 0
        xNOTcount = 0
        avyCount = [0] * nc
        avyNOTcount = [0] * nc
        for u1val in range(dParameters_.getParamsPerAtt()[u1]):
            for u2val in range(dParameters_.getParamsPerAtt()[u2]):
                for u3val in range(dParameters_.getParamsPerAtt()[u3]):
                    for u4val in range(dParameters_.getParamsPerAtt()[u4]):
                        for u5val in range(dParameters_.getParamsPerAtt()[u5]):
                            if u1val == u1valin and u2val == u2valin and u3val == u3valin and u4val == u4valin and u5val == u5valin:
                                for y in range(nc):
                                    index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y)
                                    xcount += dParameters_.getCountAtFullIndex(index)
                                    avyCount[y] += dParameters_.getCountAtFullIndex(index)
                            else:
                                for y in range(nc):
                                    index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y)
                                    xNOTcount += dParameters_.getCountAtFullIndex(index)
                                    avyNOTcount[y] += dParameters_.getCountAtFullIndex(index)
        m = 0
        for c in range(nc):
            ycount = dParameters_.getCountAtFullIndex(c)
            if avyCount[c] > 0:
                m += (avyCount[c] / N) * math.log( avyCount[c] / ( xcount/N * ycount ) ) / math.log(2)
            if avyNOTcount[c] > 0:
                m += (avyNOTcount[c] / N) * math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / math.log(2)
        return m

    @staticmethod
    def sort(unsortMap: Dict[str, float]) -> Dict[str, float]:
        sortedMapAsc = OrderedDict(sorted(unsortMap.items(), key=lambda x: x[1]))
        return sortedMapAsc

    @staticmethod
    def printMap(map: Dict[str, float]) -> None:
        print("----------------------------------------------------")
        for key, value in map.items():
            print("Key : " + key + " Value : "+ str(value))
        print("----------------------------------------------------")

    @staticmethod
    def numberOfCharInString(key: str, c: str) -> int:
        num = 0
        for i in range(len(key)):
            if key[i] == c:
                num += 1
        return num

    @staticmethod
    def getStringFromLine(val: str, delimiter: str) -> List[str]:
    parseValues = val.split(delimiter)
    valuesString = [value.strip() for value in parseValues]
    return valuesString

    @staticmethod
    def getDoubleFromLine(val: str, delimiter: str) -> List[float]:
        val = val.replace("{", "").replace("}", "")
        parseValues = val.split(delimiter)
        valuesDouble = [float(value.strip()) for value in parseValues]
        return valuesDouble

    def getIntegerFromLine(val: str, delimiter: str) -> List[int]:
        if val == "{}" or val == "":
            return []
        val = val.replace("{", "").replace("}", "")
        parseValues = val.split(delimiter)
        valuesInt = [int(value.strip()) for value in parseValues]
        return valuesInt

    def getBooleanFromLine(val: str) -> List[bool]:
        val = val.replace("{", "").replace("}", "")
        parseValues = val.split(",")
        isNumericFlag = [bool(int(value.strip())) for value in parseValues]
        return isNumericFlag

    def getIndices(key: str) -> List[int]:
        vals = [int(value.strip()) for value in key.split(":|,")]
        return vals

    def sortSet(unsortMap: Dict[Set[int], float]) -> Dict[Set[int], float]:
        sortedMapAsc = dict(sorted(unsortMap.items(), key=lambda x: x[1]))
        return sortedMapAsc

    def sortSetByComparator(unsortMap: Dict[Set[int], float], order: bool) -> Dict[Set[int], float]:
        sortedMap = dict(sorted(unsortMap.items(), key=lambda x: x[1], reverse=order))
        return sortedMap

    def addNoise(numNoiseColumns: int, sourceFile: str) -> str:
        structure = pd.read_csv(sourceFile)
        out = os.path.join(os.path.dirname(sourceFile), "trainCV-.arff")
        print("(SUtils, addNoise()) Creating File at: " +  out)
        header = ""
        header += "@relation '" + "contrieved" + "'\n\n"
        for i in range(structure.shape[1] - 1):
            header += "@attribute x" + str(i) + " { "
            for j in range(len(structure.iloc[:, i].unique())):
                if j == len(structure.iloc[:, i].unique()) - 1:
                    header += str(structure.iloc[:, i].unique()[j])
                else:
                    header += str(structure.iloc[:, i].unique()[j]) + ", "
            header += " }\n"
        for i in range(numNoiseColumns):
            header += "@attribute x" + str(i + (structure.shape[1] - 1)) + " {0, 1, 2}\n"
        classIndex = structure.shape[1] - 1
        header += "@attribute x" + str(classIndex + numNoiseColumns) + " { "
        for j in range(len(structure.iloc[:, classIndex].unique())):
            if j == len(structure.iloc[:, classIndex].unique()) - 1:
                header += str(structure.iloc[:, classIndex].unique()[j])
            else:
                header += str(structure.iloc[:, classIndex].unique()[j]) + ", "
        header += " }\n"
        header += "\n@data\n\n"
        with open(out, "w") as f:
            f.write(header)
            for i in range(structure.shape[0]):
                for u in range(structure.shape[1] - 1):
                    f.write(str(structure.iloc[i, u]) + ",")
                for j in range(numNoiseColumns):
                    f.write(str(random.randint(0, 2)) + ",")
                f.write(str(structure.iloc[i, structure.shape[1] - 1]) + "\n")
        return out

    def sampleFromNonUniformDistribution(ds: List[float], r: Any) -> int:
        rand = r.uniform(0.0, 1.0)
        chosenVal = 0
        sumProbs = ds[chosenVal]
        while rand > sumProbs:
            chosenVal += 1
            sumProbs += ds[chosenVal]
        return chosenVal

    def getArrayListDoubleMean(list: List[float]) -> float:
        return np.mean(list)

    def getArrayListDoubleVariance(list: List[float]) -> float:
        return np.var(list)

    def getArrayListDoubleMin(list: List[float]) -> float:
        return np.min(list)

    def getArrayListDoubleMax(list: List[float]) -> float:
        return np.max(list)

    def getArrayListDoubleMode(list: List[float]) -> float:
        return np.argmax(np.bincount(list))

    def ind(i, j):
        return 1 if i == j else 0

    def MEsti(freq1, freq2, numValues):
        m_MEsti = 1.0
        mEsti = (freq1 + m_MEsti / numValues) / (freq2 + m_MEsti)
        return mEsti

    def boundAndNormalizeInLogDomain(logs, maxDifference):
        boundDifferences(logs, maxDifference)
        logSum = sumInLogDomain(logs)
        for i in range(len(logs)):
            logs[i] -= logSum

    def boundDifferences(logs, maxDifference):
        maxLog = logs[0]
        for i in range(1, len(logs)):
            if maxLog < logs[i]:
                maxLog = logs[i]
        for i in range(len(logs)):
            logs[i] = logs[i] - maxLog
            if logs[i] < -maxDifference:
                logs[i] = -maxDifference

    def normalizeInLogDomain(logs):
        logSum = sumInLogDomain(logs)
        for i in range(len(logs)):
            logs[i] -= logSum

    def sumInLogDomain(logs):
        maxLog = logs[0]
        idxMax = 0
        for i in range(1, len(logs)):
            if maxLog < logs[i]:
                maxLog = logs[i]
                idxMax = i
        sum = 0
        for i in range(len(logs)):
            if i == idxMax:
                sum += 1
            else:
                sum += math.exp(logs[i] - maxLog)
        return maxLog + math.log(sum)

    def exp(logs):
        for c in range(len(logs)):
            logs[c] = math.exp(logs[c])

    def exp2(logs):
        a = np.zeros(len(logs))
        for c in range(len(logs)):
            a[c] = math.exp(logs[c])
        return a

    def log(logs):
        for c in range(len(logs)):
            logs[c] = math.log(logs[c])

    def sort(mi):
        sortedPositions = np.argsort(mi)
        n = len(mi)
        order = np.zeros(n)
        for i in range(n):
            order[i] = sortedPositions[(n-1) - i]
        return order

    def combination(N, k):
        n = 0
        num = factorial(N)
        denum1 = factorial(N - k)
        denum2 = factorial(k)
        n = (num) // (denum1 * denum2)
        return n

    def factorial(a):
        facta = 1
        for i in range(a, 0, -1):
            facta *= i
        return facta

    def NC2(a):
        count = 0
        for att1 in range(1, a):
            for att2 in range(0, att1):
                count += 1
        return count

    def NC3(a):
        count = 0
        for att1 in range(2, a):
            for att2 in range(1, att1):
                for att3 in range(0, att2):
                    count += 1
        return count

    def NC4(a):
        count = 0
        for att1 in range(3, a):
            for att2 in range(2, att1):
                for att3 in range(1, att2):
                    for att4 in range(0, att3):
                        count += 1
        return count

    def NC5(a):
        count = 0
        for att1 in range(4, a):
            for att2 in range(3, att1):
                for att3 in range(2, att2):
                    for att4 in range(1, att3):
                        for att5 in range(0, att4):
                            count += 1
        return count

    def randomize(index, n):
        random.seed()
        for i in range(len(index)):
            k = random.randint(0, n-1)
            index[i] = k

    def randomize(index):
        random.seed()
        for j in range(len(index) - 1, 0, -1):
            k = random.randint(0, j)
            temp = index[j]
            index[j] = index[k]
            index[k] = temp

    def shuffleArray(ar):
        np.random.shuffle(ar)

    def maxAbsValueInAnArray(array):
        index = 0
        max = float('-inf')
        for i in range(len(array)):
            if abs(array[i]) > max:
                max = abs(array[i])
                index = i
        return abs(array[index])

    def maxLocationInAnArray(array):
        index = 0
        max = float('-inf')
        for i in range(len(array)):
            if array[i] > max:
                max = array[i]
                index = i
        return index

    def minLocationInAnArray(array):
        index = 0
        min = float('inf')
        for i in range(len(array)):
            if array[i] < min:
                min = array[i]
                index = i
        return index

    def findMaxValueLocationInNDMatrix(results, dim):
        tempVector = np.zeros(len(results))
        for i in range(len(results)):
            tempVector[i] = results[i][dim]
        index = minLocationInAnArray(tempVector)
        return index

    def MI(contingencyMatrix):
        n = 0
        nrows = len(contingencyMatrix)
        ncols = len(contingencyMatrix[0])
        rowsSum = np.zeros(nrows)
        colsSum = np.zeros(ncols)
        for r in range(nrows):
            for c in range(ncols):
                rowsSum[r] += contingencyMatrix[r][c]
                colsSum[c] += contingencyMatrix[r][c]
                n += contingencyMatrix[r][c]
        MI = 0
        for r in range(nrows):
            if rowsSum[r] != 0:
                for c in range(ncols):
                    if colsSum[c] != 0:
                        if contingencyMatrix[r][c] > 0:
                            a = contingencyMatrix[r][c] / ( rowsSum[r]/n * colsSum[c] )
                            MI += contingencyMatrix[r][c]/n * math.log(a / math.log(2))
        return MI

    def gsquare(observed):
        n = 0
        nrows = len(observed)
        ncols = len(observed[0])
        rowsSum = np.zeros(nrows)
        colsSum = np.zeros(ncols)
        for r in range(nrows):
            for c in range(ncols):
                rowsSum[r] += observed[r][c]
                colsSum[c] += observed[r][c]
                n += observed[r][c]
        gs = 0.0
        for r in range(nrows):
            if rowsSum[r] != 0:
                for c in range(ncols):
                    if colsSum[c] != 0:
                        if observed[r][c] > 0:
                            exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n)
                            gs += 1.0*observed[r][c]*math.log(observed[r][c]/exp)
        gs *= 2.0
        return gs

    def chisquare(observed):
        n = 0
        nrows = len(observed)
        ncols = len(observed[0])
        rowsSum = np.zeros(nrows)
        colsSum = np.zeros(ncols)
        for r in range(nrows):
            for c in range(ncols):
                rowsSum[r] += observed[r][c]
                colsSum[c] += observed[r][c]
                n += observed[r][c]
        chi = 0.0
        for r in range(nrows):
            if rowsSum[r] != 0:
                for c in range(ncols):
                    if colsSum[c] != 0:
                        if observed[r][c] > 0:
                            exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n)
                            diff = observed[r][c]-exp
                            chi += diff*diff/exp
        return chi

    def inSet(array, element):
        return element in array

    def linkExist(m_Parents, m_Order, i, j):
        present = False
        if inSet(m_Parents[i], m_Order[j]):
            present = True
        if inSet(m_Parents[j], m_Order[i]):
            present = True
        return present

    def CheckForPerfectness(m_TempParents, m_Parents, m_Order):
        parents = []
        j = 0
        for j1 in range(len(m_TempParents)):
            for j2 in range(j1 + 1, len(m_TempParents)):
                if linkExist(m_Parents, m_Order, m_TempParents[j1], m_TempParents[j2]):
                    parents.append(m_TempParents[j1])
                    parents.append(m_TempParents[j2])
                    j += 2
        return None

    def labelsToProbs(labels, probs):
        count = np.zeros(len(probs))
        for i in range(len(labels)):
            count[labels[i]] += 1
        maxCount = float('-inf')
        labelIndex = -1
        for i in range(len(count)):
            if count[i] > maxCount:
                maxCount = count[i]
                labelIndex = i
        probs[labelIndex] = 1.0

    def inArray(val, arr):
        return val in arr

    def minAbsValueInAnArray(array):
        index = 0
        min = float('inf')
        for i in range(len(array)):
            if abs(array[i]) < min:
                min = abs(array[i])
                index = i
        return abs(array[index])

    def minNonZeroValueInAnArray(array):
        index = 0
        min = float('inf')
        for i in range(len(array)):
            if array[i] != 0 and array[i] < min:
                min = abs(array[i])
                index = i
        return array[index]

    def generateBaggedData(instances):
        N = instances.shape[0]
        data = np.zeros(instances.shape)
        for i in range(N):
            index = random.randint(0, N-1)
            data[i] = instances[index]
        return data

    def normalize(ds):
        sum = np.sum(ds)
        if sum > 0:
            ds /= sum

    def monotonic(results):
        diff = np.zeros(len(results) - 1)
        for i in range(len(results) - 2):
            diff[i] = results[i + 1] - results[i]
        numSadlePoints = 0
        for i in range(diff.shape[0] - 2):
            if not sameSign(diff[i + 1], diff[i]):
                numSadlePoints += 1
        if numSadlePoints > 1:
            return False
        else:
            return True

    def sameSign(a, b):
        return (a < 0) == (b < 0)

    def getTrainTestIndices(N):
        Nvalidation = 0
        if N // 10 >= 10000:
            Nvalidation = 10000
        else:
            Nvalidation = N // 10
        print("Creating Validation (CV) file of size: " + str(Nvalidation))
        indexList = []
        nvalid = 0
        while nvalid < Nvalidation:
            index = random.randint(0, N-1)
            if index not in indexList:
                indexList.append(index)
                nvalid += 1
        return indexList

    def getTrainTestInstances(cvInstances):
        instancesList = [None, None]
        instancesList[0] = cvInstances[::10]
        instancesList[1] = cvInstances[1::10]
        print("-- Train Test files created for cross-validating step size -- Train = " + str(instancesList[0].shape[0]) + ", and Test = " + str(instancesList[1].shape[0]))
        return instancesList

    def getTrainTestInstances(sourceFile, indexList, BUFFER_SIZE):
        data = datasets.load_svmlight_file(sourceFile)
        instances = data[0]
        instancesTrain = []
        instancesTest = []
        nvalidation = len(indexList)
        i = 0
        for row in instances:
            if i in indexList:
                if nvalidation % 5 == 0:
                    instancesTest.append(row)
                else:
                    instancesTrain.append(row)
                nvalidation += 1
            i += 1
        instancesList = [np.array(instancesTrain), np.array(instancesTest)]
        print("-- Train Test files created for cross-validating step size -- Train = " + str(instancesList[0].shape[0]) + ", and Test = " + str(instancesList[1].shape[0]))
        return instancesList

    def getStratifiedIndices(sourceFile, BUFFER_SIZE, ARFF_BUFFER_SIZE, S):
        res = np.zeros(0)
        instances, _ = datasets.load_svmlight_file(sourceFile)
        nc = instances.shape[1]
        classCount = np.zeros(nc)
        numToBeSelected = np.zeros(nc)
        numSelected = np.zeros(nc)
        selectionProb = np.zeros(nc)
        i = 0
        for row in instances:
            x_C = int(row[-1])
            classCount[x_C] += 1
            i += 1
        for c in range(nc):
            if classCount[c] < 50:
                numToBeSelected[c] = classCount[c] // 2
                selectionProb[c] = 0.5
            else:
                numToBeSelected[c] = (classCount[c] // 100) * S
                selectionProb[c] = S / 100
        i = 0
        for row in instances:
            x_C = int(row[-1])
            if random.random() < selectionProb[x_C]:
                res = np.append(res, i)
                numSelected[x_C] += 1
            i += 1
        print("-------------------------------------------------------------------")
        print("Class Counts = " + str(classCount))
        print("Num to be selected = " + str(numToBeSelected))
        print("Actually selected = " + str(numSelected))
        print("-------------------------------------------------------------------")
        return res
    def get_stratified_indices(S, data):
        res = set()
        with open(data, 'r') as f:
            lines = f.readlines()
            nc = int(lines[0].split()[-1])
            classCount = [0] * nc
            numToBeSelected = [0] * nc
            numSelected = [0] * nc
            selectionProb = [0] * nc
            for line in lines[1:]:
                x_C = int(line.split()[-1])
                classCount[x_C] += 1
            for c in range(nc):
                if classCount[c] < 50:
                    numToBeSelected[c] = classCount[c] // 2
                    selectionProb[c] = 0.5
                else:
                    numToBeSelected[c] = (classCount[c] // 100) * S
                    selectionProb[c] = S / 100
            lineNo = 0
            for line in lines[1:]:
                x_C = int(line.split()[-1])
                if random.random() < selectionProb[x_C]:
                    res.add(lineNo)
                    numSelected[x_C] += 1
                lineNo += 1
        print("-------------------------------------------------------------------")
        print("Class Counts =", classCount)
        print("Num to be selected =", numToBeSelected)
        print("Actually selected =", numSelected)
        print("-------------------------------------------------------------------")
        return res

    def get_train_test_instances(res, data):
        train_instances = []
        test_instances = []
        with open(data, 'r') as f:
            lines = f.readlines()
            for lineNo, line in enumerate(lines[1:]):
                if lineNo in res:
                    train_instances.append(line)
                else:
                    test_instances.append(line)
        print("-- CVInstances file created (in memory) -- Size =", len(train_instances))
        return train_instances, test_instances

    def discretize_data(data, type, numBins):
        out = "trainCV.arff"
        print("Creating File at:", out)
        with open(data, 'r') as f:
            lines = f.readlines()
            m_DiscreteInstances = []
            if type == 1:
                print("Starting MDL Discretization")
                # perform MDL Discretization
                for line in lines[1:]:
                    m_DiscreteInstances.append(line)
            elif type == 2:
                print("Starting Equal Frequency Discretization with", numBins, "bins.")
                # perform Equal Frequency Discretization
                for line in lines[1:]:
                    m_DiscreteInstances.append(line)
        with open(out, 'w') as f:
            f.write(lines[0])
            for instance in m_DiscreteInstances:
                f.write(instance)
        return out

    def normalize_data(data):
        print("Starting Normalization")
        out = "trainCV.arff"
        with open(data, 'r') as f:
            lines = f.readlines()
            m_NormalizedInstances = []
            # perform normalization
            for line in lines[1:]:
                m_NormalizedInstances.append(line)
        with open(out, 'w') as f:
            f.write(lines[0])
            for instance in m_NormalizedInstances:
                f.write(instance)
        return out

    def get_test0_indexes(N):
        res = set()
        nLines = 0
        for i in range(N):
            if random.random() < 0.5:
                res.add(nLines)
            nLines += 1
        expectedNLines = nLines // 2 if nLines % 2 == 0 else nLines // 2 + 1
        actualNLines = len(res)
        if actualNLines < expectedNLines:
            while actualNLines < expectedNLines:
                chosen = random.randint(0, nLines-1)
                if chosen not in res:
                    res.add(chosen)
                    actualNLines += 1
        elif actualNLines > expectedNLines:
            while actualNLines > expectedNLines:
                chosen = random.randint(0, nLines-1)
                if chosen in res:
                    res.remove(chosen)
                    actualNLines -= 1
        return res

    def create_train_tmp_file(structure, trainIndexes):
        out = "trainCV.arff"
        print("Creating File at:", out)
        with open(Globals.getSOURCEFILE(), 'r') as f:
            lines = f.readlines()
            CVInstances = []
            for lineNo, line in enumerate(lines[1:]):
                if lineNo in trainIndexes:
                    CVInstances.append(line)
        with open(out, 'w') as f:
            f.write(lines[0])
            for instance in CVInstances:
                f.write(instance)
        return out

    def set_structure():
        with open(Globals.getSOURCEFILE(), 'r') as f:
            lines = f.readlines()
            n = int(lines[0].split()[-1])
            nc = int(lines[0].split()[-2])
            isNumericTrue = [False] * n
            paramsPerAtt = [0] * n
            for u, line in enumerate(lines[1:]):
                if line.split()[-1] == 'numeric':
                    isNumericTrue[u] = True
                    paramsPerAtt[u] = 1
                else:
                    paramsPerAtt[u] = len(line.split()[-1].split(','))
        return n, nc, isNumericTrue, paramsPerAtt

    def determine_num_data():
        with open(Globals.getSOURCEFILE(), 'r') as f:
            lines = f.readlines()
            return len(lines) - 1

    def combine_indexes(indexes, fold):
        foldIndexes = set()
        for i in range(len(indexes)):
            if i != fold:
                foldIndexes.update(indexes[i])
        return foldIndexes

    def get_indexes(indexes):
        numFolds = Globals.getNumFolds()
        N = Globals.getNumberInstances()
        expectedNumberInEachBin = N // numFolds
        leftOvers = N % numFolds
        for i in range(N):
            bin = get_bin(numFolds)
            indexes[bin].add(i)
        movers = []
        for i in range(numFolds):
            diff = len(indexes[i]) - expectedNumberInEachBin
            if diff > 0:
                while len(indexes[i]) != expectedNumberInEachBin:
                    j = random.randint(0, N-1)
                    if j in indexes[i]:
                        movers.append(j)
                        indexes[i].remove(j)
        for i in range(numFolds):
            diff = len(indexes[i]) - expectedNumberInEachBin
            if diff < 0:
                while len(indexes[i]) != expectedNumberInEachBin:
                    if not movers:
                        break
                    j = movers.pop()
                    indexes[i].add(j)

    def get_bin(numFolds):
        randProbs = np.random.rand(numFolds)
        return np.argmax(randProbs)

    def get_results(probs, x_C, nc):
        results = [0] * 4
        rmse = 0
        loss = 0
        neglogloss = 0
        pred = -1
        bestProb = float('-inf')
        for y in range(nc):
            if not math.isnan(probs[y]):
                if probs[y] > bestProb:
                    pred = y
                    bestProb = probs[y]
                rmse += (1 / nc * (probs[y] - (1 if y == x_C else 0)) ** 2)
            else:
                pass
        if pred != x_C:
            loss = 1
        neglogloss = -math.log(probs[x_C])
        results[0] = rmse
        results[1] = loss
        results[2] = neglogloss
        results[3] = pred
        return results

    def randomize_training_file():
        with open(Globals.getSOURCEFILE(), 'r') as f:
            lines = f.readlines()
            N = len(lines) - 1
            randIndices = np.random.permutation(N)
            cache = {}
            out = "trainCV.arff"
            print("(randomizeTrainingFile): Creating File at:", out)
            with open(out, 'w') as f:
                f.write(lines[0])
                for i, line in enumerate(lines[1:]):
                    cache[i] = line
                    lookingFor = randIndices[i]
                    if i == lookingFor:
                        for j in range(i, len(randIndices)):
                            extract = randIndices[j]
                            InstanceC = cache[extract]
                            f.write(InstanceC)
                            del cache[extract]
        print("New Source File after Randomization is:", out)
        return out

    def delete_source_file():
        os.remove(Globals.getSOURCEFILE())

    def is_prime(n):
        if n < 2:
            return False
        elif n == 2:
            return True
        elif n % 2 == 0:
            return False
        else:
            sqrtN = int(math.sqrt(n))
            for i in range(3, sqrtN+1, 2):
                if n % i == 0:
                    print(n, "is divisible by:", i)
                    return False
        return True

    def get_previous_prime(n):
        pn = 1
        for i in range(n-1, 1, -1):
            if is_prime(i):
                return i
        return pn


