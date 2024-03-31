import math
from datastructure import xxyDist
from datastructure import xyDist

class CorrelationMeasures:
    
    @staticmethod
    def findMST(numNode, weight, m_Parents):
        inTree = [0] * numNode
        distance = [0] * numNode
        
        for i in range(numNode):
            inTree[i] = -1
        
        inTree[0] = 1
        
        for i in range(1, numNode):
            distance[i] = weight[0][i]
            m_Parents[i] = 0
        
        for treeSize in range(1, numNode):
            max = -1
            for i in range(numNode):
                if inTree[i] != 1:
                    if max == -1 or distance[max] < distance[i]:
                        max = i
            
            inTree[max] = 1
            
            for i in range(numNode):
                if inTree[i] == 1:
                    continue
                if distance[i] < weight[max][i]:
                    distance[i] = weight[max][i]
                    m_Parents[i] = max
    
    @staticmethod
    def getMIbasedParent(xxyDist_):
        nc = xxyDist_.getNoClasses()
        n = xxyDist_.getNoAtts()
        paramsPerAtt = [0] * n
        for u in range(n):
            paramsPerAtt[u] = xxyDist_.getNoValues(u)
        
        m_Parents = [-1] * n
        
        m_CondiMutualInfo = [[0] * n for _ in range(n)]
        
        for u1 in range(n):
            for u2 in range(u1, -1, -1):
                if u1 == u2:
                    continue
                m_CondiMutualInfo[u1][u2] = 0
                for u1val in range(paramsPerAtt[u1]):
                    for u2val in range(paramsPerAtt[u2]):
                        for c in range(nc):
                            mi = 0
                            
                            mi = xxyDist_.jointP(u1, u1val, u2, u2val, c) * \
                                math.log((xxyDist_.jointP(u1, u1val, u2, u2val, c) * xxyDist_.xyDist_.p(c)) / \
                                         (xxyDist_.xyDist_.jointP(u1, u1val, c) * xxyDist_.xyDist_.jointP(u2, u2val, c)))
                            
                            m_CondiMutualInfo[u1][u2] += mi
                            m_CondiMutualInfo[u2][u1] += mi
        
        MI = [0] * n
        for att1 in range(n):
            for att2 in range(n):
                MI[att1] += m_CondiMutualInfo[att1][att2]
        
        CorrelationMeasures.findMST(n, m_CondiMutualInfo, m_Parents)
        return m_Parents
    
    @staticmethod
    def getMutualInformation(xyDist_, mi):
        nc = xyDist_.getNoClasses()
        n = xyDist_.getNoAtts()
        N = xyDist_.getNoData()
        paramsPerAtt = [0] * n
        for u in range(n):
            paramsPerAtt[u] = xyDist_.getNoValues(u)
        
        for u in range(n):
            m = 0
            for uval in range(paramsPerAtt[u]):
                for y in range(nc):
                    avyCount = xyDist_.getCount(u, uval, y)
                    if avyCount > 0:
                        m += (avyCount / N) * math.log(avyCount / (xyDist_.getCount(u, uval) / N * xyDist_.getClassCount(y))) / math.log(2)
            
            mi[u] = m
    
    @staticmethod
    def getCondMutualInf(xxyDist_, cmi):
        nc = xxyDist_.getNoClasses()
        n = xxyDist_.getNoAtts()
        N = xxyDist_.xyDist_.getNoData()
        paramsPerAtt = [0] * n
        for u in range(n):
            paramsPerAtt[u] = xxyDist_.getNoValues(u)
        
        for u1 in range(1, n):
            for u2 in range(u1):
                mi = 0
                for u1val in range(paramsPerAtt[u1]):
                    for u2val in range(paramsPerAtt[u2]):
                        for c in range(nc):
                            avvyCount = xxyDist_.getCount(u1, u1val, u2, u2val, c)
                            if avvyCount > 0:
                                a = avvyCount
                                b = xxyDist_.xyDist_.getClassCount(c)
                                d = xxyDist_.xyDist_.getCount(u1, u1val, c)
                                e = xxyDist_.xyDist_.getCount(u2, u2val, c)
                                
                                mitemp = (a / N) * math.log((a * b) / (d * e)) / math.log(2)
                                mi += mitemp
                
                cmi[u1][u2] = mi
                cmi[u2][u1] = mi


