import jnius_config
jnius_config.add_options('-Xmx2G')
jnius_config.set_classpath('am-tools.jar')
from jnius import autoclass
import random
import time
import numpy


if __name__ == "__main__":
    start_time = time.time()
    k = 100000
    Rule = autoclass('de.up.ling.irtg.automata.Rule')
    ConcreteTreeAutomaton = autoclass('de.up.ling.irtg.automata.ConcreteTreeAutomaton')
    PyjniusHelper = autoclass('de.saar.coli.amtools.decomposition.PyjniusHelper')
    auto = ConcreteTreeAutomaton()
    print("time for init:")
    print(time.time()-start_time)
    start_time = time.time()
    rules = [auto.createRule(1,2,[3,4], 1) for i in range(k)]
    weights = [random.uniform(0, 1) for i in range(k)]
    print("time for array creation:")
    print(time.time()-start_time)
    start_time = time.time()
    for i in range(k):
        rules[i].setWeight(weights[i]) #random.uniform(0, 1))
        # print(rule.getWeight())
    print("time for python i:")
    print(time.time()-start_time)
    start_time = time.time()

    for rule, weight in zip(rules, weights):
        rule.setWeight(weight)
    print("time for python zip:")
    print(time.time()-start_time)
    start_time = time.time()

    java_time = PyjniusHelper.assignRuleWeights(rules, weights)

    print("time for java rule array:")
    print(time.time()-start_time)
    print("as measured in java (ms):")
    print(java_time)

    rule = rules[0]
    start_time = time.time()

    java_time = PyjniusHelper.assignRuleWeights(rule, weights)

    print("time for java single rule:")
    print(time.time() - start_time)
    print("as measured in java (ms):")
    print(java_time)

    start_time = time.time()
    newWeights = PyjniusHelper.assignRuleWeightsReturnWeights(rule, weights)
    print("time for java single rule with returned weights:")
    print(time.time() - start_time)
    print(newWeights[0])
    print(len(newWeights))
    #print(newWeights)

    array = numpy.random.rand(k)
    print(array)
    start_time = time.time()
    list_from_array = array.tolist()
    print("time for numpy array to python list conversion:")
    print(time.time() - start_time)
    #print(list_from_array)
    newWeights = PyjniusHelper.assignRuleWeightsReturnWeights(rule, list_from_array)
    print(array[0])
    print(newWeights[0])
