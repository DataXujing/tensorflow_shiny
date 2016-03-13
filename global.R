##Global

library(shiny)
library(plotly)
library(rPython)
library(Metrics)

python.exec("
import sys
sys.argv = ['']
from numpy import array, sin, cos, pi
import numpy as np
from random import random
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
frequency1 = 311
frequency2 = 201
")


