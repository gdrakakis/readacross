#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
import sklearn
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
import io
from io import BytesIO
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image ## Hide for production

app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for Read Across
"""
def getJsonContentsRA (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
        readAcrossURIs = parameters.get("readAcrossURIs", None) # nanoparticles for readAcross

        variables = dataEntry[0]["values"].keys() 
        variables.sort()  # NP features including predictionFeature

        datapoints =[] # list of nanoparticle feature vectors not for readacross
        read_across_datapoints = [] #list of readacross nanoparticle feature vectors

        nanoparticles=[] # nanoparticles not in readAcrossURIs list
        target_variable_values = [] # predictionFeature values

        for i in range(len(dataEntry)-len(readAcrossURIs)):
            datapoints.append([])

        for i in range(len(readAcrossURIs)):
            read_across_datapoints.append([])


        counter = 0
        RAcounter = 0
        for i in range(len(dataEntry)):

            if dataEntry[i]["compound"].get("URI") not in readAcrossURIs:
                nanoparticles.append(dataEntry[i]["compound"].get("URI"))
                for j in variables:
                    if j == predictionFeature:
                        target_variable_values.append(dataEntry[i]["values"].get(j))
                    else:
                        datapoints[counter].append(dataEntry[i]["values"].get(j))
                counter+=1
            else:
                for j in variables:
                    if j != predictionFeature:
                        read_across_datapoints[RAcounter].append(dataEntry[i]["values"].get(j))
                RAcounter+=1

        variables.remove(predictionFeature) # NP features

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
    #print len(nanoparticles), len(read_across_datapoints)
    #print readAcrossURIs, read_across_datapoints
    return variables, datapoints, read_across_datapoints, predictionFeature, target_variable_values, byteify(readAcrossURIs), nanoparticles

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

"""
    [[],[]]  Matrix to dictionary for Nearest Neighboura
"""
def mat2dicNN(matrix, name):
    myDict = {}
    for i in range (len (matrix[0])):
        myDict[name + " NN_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)

"""
    [[],[]]  Matrix to dictionary 
"""
def mat2dic(matrix):
    myDict = {}
    for i in range (len (matrix)):
        myDict["Row_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)

"""
    [[]]  Matrix to dictionary Single Row
"""
def mat2dicSingle(matrix):
    myDict = {}
    myDict["Row_1"] = matrix
    return byteify(myDict)

"""
    Normaliser
"""
def manual_norm(myTable, myMax, myMin):
    if myMax>myMin:
        for i in range (len(myTable)):
            myTable[i] = (myTable[i]-myMin)/(myMax-myMin)
    else:
        for i in range (len(myTable)):
            myTable[i] = 0
    return myTable

"""
    Distances
"""
def distances (read_across_datapoints, datapoints, variables, readAcrossURIs, nanoparticles):
    #print read_across_datapoints
    #for i in range (len (readAcrossURIs)):
    #    for j in range (len (nanoparticles)):
    #        for k in range (len (variables)):
    #            ...
    
    datapoints_transposed = map(list, zip(*datapoints)) 
    RA_datapoints_transposed = map(list, zip(*read_across_datapoints)) 
    
    for i in range (len(datapoints_transposed)):
        max4norm = numpy.max(datapoints_transposed[i])
        min4norm = numpy.min(datapoints_transposed[i])

        datapoints_transposed[i] = manual_norm(datapoints_transposed[i], max4norm, min4norm)
        RA_datapoints_transposed[i] = manual_norm(RA_datapoints_transposed[i], max4norm, min4norm)

    #print RA_datapoints_transposed[0]
    #print datapoints_transposed[0]

    term1 = []
    term2 = []
    for i in range (len(variables)):
        #term1.append(numpy.min(datapoints_transposed))
        #term2.append(numpy.max(datapoints_transposed))
        term1.append(0)
        term2.append(1)

    #transpose back
    datapoints_norm = map(list, zip(*datapoints_transposed)) 
    RA_datapoints_norm = map(list, zip(*RA_datapoints_transposed)) 

    #print numpy.max(RA_datapoints_norm)
    #print numpy.max(datapoints_norm)

    #for i in range (len(datapoints)):
    #    datapoints[i] = manual_norm(datapoints[i], max4norm, min4norm)
    #for i in range (len(read_across_datapoints)):
    #    read_across_datapoints[i] = manual_norm(read_across_datapoints[i], max4norm, min4norm)


    max_eucl_dist = euclidean_distances(term1, term2)
    eucl_dist = euclidean_distances(RA_datapoints_norm, datapoints_norm)
    eucl_dist = numpy.array(eucl_dist)
    eucl_dist = eucl_dist/max_eucl_dist
    eucl_dist = numpy.round(eucl_dist,4)

    np_plus_eucl = []
    for i in range (len(readAcrossURIs)):
        np_plus_eucl.append([nanoparticles, eucl_dist[i]]) 
    #print np_plus_eucl

    eucl_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_eucl[i][0], np_plus_eucl[i][1]
        np = zip (np_plus_eucl[i][1], np_plus_eucl[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        eucl_sorted.append([np_sorted, dist_sorted])
    #print "\n\nSorted\n\n", eucl_sorted
    ## [ [ [names] [scores] ] [ [N] [S] ]]
    ##       00      01          10  11    


    #eucl_transposed = map(list, zip(*eucl_sorted)) 
    eucl_dict = {} # []
    for i in range (len(readAcrossURIs)):
        #print "\n HERE \n ", eucl_sorted[i]
        #eucl_dict.append(mat2dicNN(eucl_sorted[i], readAcrossURIs[i])) #
        for j in range (len (eucl_sorted[i][0])):
            eucl_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [eucl_sorted[i][0][j], eucl_sorted[i][1][j]]
    eucl_dict = byteify(eucl_dict)
    #print "\n\nDict\n\n",eucl_dict

    max_manh_dist = metrics.pairwise.manhattan_distances(term1, term2)
    manh_dist = metrics.pairwise.manhattan_distances(RA_datapoints_norm, datapoints_norm)
    manh_dist = numpy.array(manh_dist)
    manh_dist = manh_dist/max_manh_dist
    manh_dist = numpy.round(manh_dist,4)

    np_plus_manh = []
    for i in range (len(readAcrossURIs)):
        np_plus_manh.append([nanoparticles, manh_dist[i]]) 

    manh_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_manh[i][0], np_plus_manh[i][1]
        np = zip (np_plus_manh[i][1], np_plus_manh[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        manh_sorted.append([np_sorted, dist_sorted])
    #print manh_sorted

    manh_dict = {}
    for i in range (len(readAcrossURIs)):
        #manh_dict.append(mat2dicNN(manh_sorted[i], readAcrossURIs[i]))
        for j in range (len (manh_sorted[i][0])):
            manh_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [manh_sorted[i][0][j], manh_sorted[i][1][j]]
    manh_dict = byteify(manh_dict)

    ensemble_dist = (eucl_dist + manh_dist)/2
    #print "Eucl.: ", eucl_dist, "\n Manh.: ", manh_dist,"\n Ens.: ", ensemble_dist

    np_plus_ens = []
    for i in range (len(readAcrossURIs)):
        np_plus_ens.append([nanoparticles, ensemble_dist[i]]) 

    ens_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_ens[i][0], np_plus_ens[i][1]
        np = zip (np_plus_ens[i][1], np_plus_ens[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        ens_sorted.append([np_sorted, dist_sorted])
    #print ens_sorted

    ens_dict = {}
    for i in range (len(readAcrossURIs)):
        #ens_dict.append(mat2dicNN(ens_sorted[i], readAcrossURIs[i]))
        for j in range (len (ens_sorted[i][0])):
            ens_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [ens_sorted[i][0][j], ens_sorted[i][1][j]]
    ens_dict = byteify(ens_dict)

    ### PLOT PCA
    pcafig = plt.figure()
    ax = pcafig.add_subplot(111, projection='3d')

    pca = decomposition.PCA(n_components=3)
    pca.fit(datapoints_norm)
    dt = pca.transform(datapoints_norm)
    ax.scatter(dt[:,0], dt[:,1], dt[:,2], c='r',  label = 'Original Values')

    RA_dt = pca.transform(RA_datapoints_norm)
    ax.scatter(RA_dt[:,0], RA_dt[:,1], RA_dt[:,2], c='b', label = 'Read Across Values')

    ax.set_xlabel("1st Principal Component") 
    ax.set_ylabel("2nd Principal Component")
    ax.set_zlabel("3rd Principal Component")
    ax.set_title("3D Projection of Datapoints")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    #plt.tight_layout()
    #plt.show() #HIDE show on production

    figfile = BytesIO()
    pcafig.savefig(figfile, dpi=300, format='png', bbox_inches='tight') #bbox_inches='tight'
    figfile.seek(0) 
    pcafig_encoded = base64.b64encode(figfile.getvalue())    
    
    return eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded

"""
    Predict
"""
def RA_predict(euclidean, manhattan, ensemble, name, predictionFeature, nano2value):
    #print euclidean[0] # names of np
    #print euclidean[1] # dist values
    #print nano2value

    eu_score = 0
    ma_score = 0
    en_score = 0

    eu_div = 0
    ma_div = 0
    en_div = 0

    for i in range (len(euclidean[0])):
        if euclidean[1][i] < 1:
            eu_score += (1 - euclidean[1][i])*(nano2value[euclidean[0][i]]) #just the name
            eu_div += 1 - euclidean[1][i]
        if manhattan[1][i] < 1:
            ma_score += (1 - manhattan[1][i])*(nano2value[euclidean[0][i]]) #just the name
            ma_div += 1 - manhattan[1][i]
        if ensemble[1][i] < 1:
            en_score += (1 - ensemble[1][i])*(nano2value[euclidean[0][i]]) #just the name
            en_div += 1 - ensemble[1][i]
    eu_score = eu_score/eu_div
    ma_score = ma_score/ma_div
    en_score = en_score/en_div
    #print eu_score
    return [name, round(eu_score,2)], [name, round(ma_score,2)], [name, round(en_score,2)]


"""
    Pseudo AD
"""
def RA_applicability(euclidean, manhattan, ensemble, name):
    eu_score = 0
    ma_score = 0
    en_score = 0
    for i in range (len(euclidean[1])): # list of vals
        if euclidean[1][i] < 0.4:
            eu_score +=1
        if manhattan[1][i] < 0.33:
            ma_score +=1
        if ensemble[1][i] < 0.36:
            en_score +=1
    eu_score = eu_score/len(euclidean[1])
    ma_score = ma_score/len(euclidean[1])
    en_score = en_score/len(euclidean[1])
    #RA_appl = [["Euclidean", eu_score], ["Manhattan", ma_score], ["Ensemble", en_score]]
    #return ["Euclidean", eu_score], ["Manhattan", ma_score], ["Ensemble", en_score]
    return [name, eu_score], [name, ma_score], [name, en_score]
    
    

@app.route('/pws/readacross', methods = ['POST'])
def create_task_readacross():

    if not request.json:
        abort(400)

    variables, datapoints, read_across_datapoints, predictionFeature, target_variable_values, readAcrossURIs, nanoparticles = getJsonContentsRA(request.json)
    nano2value = {}
    for i in range (len(nanoparticles)):
        nano2value[nanoparticles[i]] = target_variable_values[i]
    eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, nanoparticles)
    
    eucl_predictions = []
    manh_predictions = [] 
    ens_predictions = []

    eucl_applicability = []
    manh_applicability = [] 
    ens_applicability = []
    for i in range (len(readAcrossURIs)):
        #dict version
        #eucl_predictions[readAcrossURIs[i]], manh_predictions[readAcrossURIs[i]], ens_predictions[readAcrossURIs[i]] = RA_predict(eucl_sorted[i], manh_sorted[i], ens_sorted[i])
        eu,ma,en = RA_predict(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], predictionFeature, nano2value)
        eucl_predictions.append(eu)
        manh_predictions.append(ma)
        ens_predictions.append(en)
        
        eu,ma,en = RA_applicability(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i])
        eucl_applicability.append(eu)
        manh_applicability.append(ma)
        ens_applicability.append(en)

    #print eucl_predictions, manh_predictions, ens_predictions
    #print eucl_applicability, manh_applicability, ens_applicability


    if len (eucl_predictions) > 1:
        # predictions
        eucl_predictions_transposed = map(list, zip(*eucl_predictions)) 
        #print "\n\n\n", eucl_predictions, eucl_predictions_transposed,"\n\n\n"
        eucl_pred_dict = mat2dic(eucl_predictions_transposed)
        #print eucl_pred_dict

        manh_predictions_transposed = map(list, zip(*manh_predictions)) 
        manh_pred_dict = mat2dic(manh_predictions_transposed)
        #print manh_pred_dict

        ens_predictions_transposed = map(list, zip(*ens_predictions)) 
        ens_pred_dict = mat2dic(ens_predictions_transposed)
        #print ens_pred_dict

        # applicability
        eucl_applicability_transposed = map(list, zip(*eucl_applicability)) 
        eucl_appl_dict = mat2dic(eucl_applicability_transposed)
        #print eucl_appl_dict

        manh_applicability_transposed = map(list, zip(*manh_applicability)) 
        manh_appl_dict = mat2dic(manh_applicability_transposed)
        #print manh_appl_dict

        ens_applicability_transposed = map(list, zip(*ens_applicability)) 
        ens_appl_dict = mat2dic(ens_applicability_transposed)
        #print ens_appl_dict
    else: 
        eucl_pred_dict = mat2dicSingle(eucl_predictions[0])
        manh_pred_dict = mat2dicSingle(manh_predictions[0])
        ens_pred_dict = mat2dicSingle(ens_predictions[0])
        eucl_appl_dict = mat2dicSingle(eucl_applicability[0])
        manh_appl_dict = mat2dicSingle(manh_applicability[0])
        ens_appl_dict = mat2dicSingle(ens_applicability[0])
    #print eucl_pred_dict

    task = {
        "singleCalculations": {
                               "Euclidean Cut-off" : 0.4,
                               "Manhattan Cut-off" : 0.33,
                               "Ensemble Cut-off" : 0.36
                              },
        "arrayCalculations": {
                               "Predictions based on Euclidean Distances":
                                   {"colNames": ["Nanoparticle", "Prediction"],
                                    "values": eucl_pred_dict
                                   },
                               "Applicability Domain for Euclidean Distances":
                                   {"colNames": ["Nanoparticle", "AD Value"],
                                    "values": eucl_appl_dict
                                   },
                               "Nearest Neighbour based on Euclidean Distances":
                                   {"colNames": ["Nanoparticle", "Distance"],
                                    "values": eucl_dict
                                   },
                               "Predictions based on Manhattan Distances":
                                   {"colNames": ["Nanoparticle", "Prediction"],
                                    "values": manh_pred_dict
                                   },
                               "Applicability Domain for Manhattan Distances":
                                   {"colNames": ["Nanoparticle", "AD Value"],
                                    "values": manh_appl_dict
                                   },
                               "Nearest Neighbour based on Manhattan Distances":
                                   {"colNames": ["Nanoparticle", "Distance"],
                                    "values": manh_dict
                                   },
                               "Predictions based on Ensemble Distances":
                                   {"colNames": ["Nanoparticle", "Prediction"],
                                    "values": ens_pred_dict
                                   },
                               "Applicability Domain for Ensemble Distances":
                                   {"colNames": ["Nanoparticle", "AD Value"],
                                    "values": ens_appl_dict
                                   },
                               "Nearest Neighbour based on Ensemble Distances":
                                   {"colNames": ["Nanoparticle", "Distance"],
                                    "values": ens_dict
                                   }
                             },
        "figures": {
                   "PCA of datapoints vs. Read-Across" : pcafig_encoded
                   }
        }

    #fff = open("C:/Python27/delete123.txt", "w")
    #fff.writelines(str(task))
    #fff.close 
    #task = {}
    jsonOutput = jsonify( task )
    
    return jsonOutput, 201 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port = 5000, debug = True)

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/readacross.json http://localhost:5000/pws/readacross
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/readacross.json http://localhost:5000/pws/readacross
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python readacross.py