rm(list = ls())

library(reticulate)
library(metabCombiner)
library(ggplot2)
library(doParallel)
library(tidyverse)
library(dplyr)

np <- import("numpy")

Run_one_config = function(o,sigM,sigRT,sigFI,d,n,binGap, norm = FALSE){

  ### Runs metabCombiner for one [overlap, m/z noise, RT noise, FI noise, RT drift shape, simulation number] configuration
  ### Resulting coupling matrix is saved as a .npy file in 'RES/couple_['configuration'].npy'

  
  config = c(o,sigM,sigRT,sigFI,d,n)
  
  overlap = toString(o)
  sigmM = toString(sigM)
  sigmRT = toString(sigRT)
  sigmFI = toString(sigFI)
  drift = toString(d)
  N = toString(n)
  
  conf = paste0(overlap,", ",sigmM,", ",sigmRT,", ",sigmFI,", '",drift,"', ",N)

  # Load the data, generated with simulate_data.py and stored in a DATA folder
  path1 = paste0('DATA/DATA1_[',conf,'].npy')
  path2 = paste0('DATA/DATA2_[',conf,'].npy')
  
  pathT = paste0('DATA/TRUE_[',conf,'].npy')
  
  binGap = binGap
  
  # Datasets formatting
  
  mat1 <- np$load(path1, allow_pickle = TRUE)
  mat2 <- np$load(path2, allow_pickle = TRUE)
  true <- np$load(pathT, allow_pickle = TRUE)
  
  dataset1 <- as.data.frame(t(mat1))
  dataset2 <- as.data.frame(t(mat2))
  
  dataset1 = dataset1 %>% dplyr::rename(tracker = V1, mz = V2, rt = V3)
  dataset2 = dataset2 %>% dplyr::rename(tracker = V1, mz = V2, rt = V3)
  
  if(norm){
    dataset1[,-c(1,2,3)] = apply(dataset1[ , -c(1,2,3)], 2, function(x) as.numeric(x))
    dataset2[,-c(1,2,3)] = apply(dataset2[ , -c(1,2,3)], 2, function(x) as.numeric(x))
    dataset1[,-c(1,2,3)] = t(scale(t(dataset1[,-c(1,2,3)])))
    dataset2[,-c(1,2,3)] = t(scale(t(dataset2[,-c(1,2,3)])))}
  
  dataset1$id <- rownames(dataset1)
  dataset2$id <- rownames(dataset2)
  
  # Incorporate prior knowledge
  
  npairs = 100
  pairs = which(true != 0, arr.ind = TRUE)
  prior = pairs[sample(nrow(pairs),npairs,replace = FALSE),]
  
  # incorporate prior knowledge in nomenclature
  for (i in 1:nrow(dataset1)){dataset1$id[i] <- paste0('DS1_',dataset1$id[i])}
  for (i in 1:nrow(dataset2)){dataset2$id[i] <- paste0('DS2_',dataset2$id[i])}
  
  for (j in 1:nrow(prior)){
    id1 = prior[j,1]
    id2 = prior[j,2]
    dataset1$id[id1] <- paste0('P_',toString(j))
    dataset2$id[id2] <- paste0('P_',toString(j))
  }
  
  # Run MetabCombiner
  data1 = metabData(dataset1, mz = "mz", rt = "rt", id = "id", adduct = NULL, samples = "V", extra = NULL)
  data2 = metabData(dataset2, mz = "mz", rt = "rt", id = "id", adduct = NULL, samples = "V", extra = NULL)
  data.combined = metabCombiner(xdata = data1, ydata = data2, binGap = binGap)
  data.report = combinedTable(data.combined)
  
  data.combined = selectAnchors(data.combined, useID = FALSE, windx = 0.03, windy = 0.03, tolmz = 0.003, tolQ = 0.3)
  set.seed(100)
  
  # Estimate RT drift without using prior knowledge
  data.combined = fit_gam(data.combined, useID = FALSE, k = seq(10, 20, 2),
                          iterFilter = 2, ratio = 2, frac = 0.5, bs = "bs",
                          family = "scat", weights = 1, method = "REML",
                          optimizer = "newton")
  
  plot(data.combined, fit = "gam", main = "Example Fit", xlab = "data1",
        ylab = "data2", pch = 19, lcol = "red", pcol = "black")
  
  # Set scores using prior knowledge
  scores = evaluateParams(data.combined, A = seq(60, 150, by = 10), B = seq(6, 20), C = seq(0.1, 1 ,0.1), fit = "gam",
                          usePPM = FALSE, minScore = 0.7, penalty = 10, groups = NULL)
  
  scores = scores[which.max(scores$score),]
  
  data.combined = calcScores(data.combined, A = scores$A, B = scores$B, C = scores$C, fit = "gam", usePPM = FALSE, groups = NULL)
  
  data.report = combinedTable(data.combined)
  data.report = labelRows(data.report, maxRankX = 3,maxRankY = 3, minScore = 0.5, 
                          conflict = 0.1, method = "score", balanced = TRUE, remove = FALSE)
  
  RES = data.report
  RES_filtered = RES[(RES$labels != 'REMOVE'),]
  
  # Reformat the output of metabCombiner into a coupling matrix
  coupling = matrix(0, nrow = length(dataset1$mz), ncol = length(dataset2$mz))
  rownames(coupling) <- dataset1$id
  colnames(coupling) <- dataset2$id
  for(l in 1:nrow(RES_filtered)){
    id1 = RES_filtered$idx[l]
    id2 = RES_filtered$idy[l]
    coupling[id1,id2] = RES_filtered$score[l]
  }
  filename = paste0('RES/couple_[',toString(config),'].npy')
  np$save(filename, coupling)
  
  return()
}

### Run the previous function for each configuration of interest and store the results in a RES folder
                                 
overlap = c(0.25, 0.5, 0.75)
sigmasM = c(0.01)
sigmasRT = c(0.2,0.5,1)
sigmasFI = c(0.1, 0.5, 1)
drift = c('smooth')
Nsample = seq(from = 1, to = 20, by = 1)

nconf = length(overlap)*length(sigmasM)*length(sigmasRT)*length(sigmasFI)*length(Nsample)*length(drift)
CONF = matrix(0, nrow = nconf, ncol = 6)

#Run_one_config(0.75, 0, 0.01, 'hetero', 'smooth', 1, 0.01)

i = 0
for(o in overlap){
  for(sigM in sigmasM){
    for(sigRT in sigmasRT){
      for(sigFI in sigmasFI){
          for (d in drift){
            for(n in Nsample){
              i = i+1
              CONF[i,] = c(o,sigM,sigRT,sigFI,d,n)
          }
        }
      }
    }
  }
}

foreach(i = 1:nrow(CONF))%do%{
    o = CONF[i,1]
    sigM = CONF[i,2]
    sigRT = CONF[i,3]
    sigFI = CONF[i,4]
    d = CONF[i,5]
    n = CONF[i,6]
    Run_one_config(o, sigM, sigRT, sigFI, d, n, 0.01)
}
