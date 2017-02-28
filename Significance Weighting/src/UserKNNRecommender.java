package net.librec.recommender.cf;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.util.*;
import java.util.Map.Entry;

@ModelData({"isRanking", "knn", "userMappingData", "itemMappingData", "userMeans", "trainMatrix", "similarityMatrix"})
public class UserKNNRecommender extends AbstractRecommender {
    private int knn;
    private int beta;
    private DenseVector userMeans;
    private SymmMatrix similarityMatrix;
    private List<Map.Entry<Integer, Double>>[] userSimilarityList;

    /**
     * @see net.librec.recommender.AbstractRecommender#setup()
     */

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        System.out.println(context.getSimilarity());
        knn = conf.getInt("rec.neighbors.knn.number");
        beta = conf.getInt("rec.similarity.beta"); // read in beta from config file
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
        getWeightedSimMat(super.trainMatrix, beta); // implementation of significance weighting
    }


    private void getWeightedSimMat(SparseMatrix train, int beta) {
        /*
          This function takes in the training matrix (a sparse matrix) and the coefficient beta
          for significance weighting. It works as following:
          a) find and store rated items' indices for each user into a set
          b) count the number of overlapped items for each pair of users (no repetition)
          c) calculate the coefficient for significance weight ( min(overlap_amt, beta)/beta )
             and update the similarity matrix
        */

        // a) find and store rated items into  a set
        for (int userIdx = 1; userIdx < numUsers; userIdx++) {                    // for each user
            HashSet ratedSet1 = new HashSet<>(train.getColumns(userIdx));
            for (int compUserIdx = 0; compUserIdx < userIdx; compUserIdx++) {    // for each user before him/her
                HashSet ratedSet2 = new HashSet<>(train.getColumns(compUserIdx));

                // b) find the intersection of these two sets
                ratedSet2.retainAll(ratedSet1);
                int count = ratedSet2.size();

                // c) calculate the coefficient & update similarity matrix
                if (count > 0 && count < beta) {                             // only update overlaps less than beta
                    double coe = (double) count/ (double) beta;
                    double temp = similarityMatrix.get(userIdx, compUserIdx) * coe;
                    similarityMatrix.set(userIdx, compUserIdx, temp);
                    similarityMatrix.set(compUserIdx, userIdx, temp);
                }
            }
        }
    }


    /**
     * @see net.librec.recommender.AbstractRecommender#trainModel()
     */

    @Override
    protected void trainModel() throws LibrecException {
        userMeans = new DenseVector(numUsers);
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            SparseVector userRatingVector = trainMatrix.row(userIdx);
            userMeans.set(userIdx, userRatingVector.getCount() > 0 ? userRatingVector.mean() : globalMean);
        }
    }

    /**
     * @see net.librec.recommender.AbstractRecommender#predict(int, int)
     */

    @Override
    public double predict(int userIdx, int itemIdx) throws LibrecException {
        //create userSimilarityList if not exists
        if (!(null != userSimilarityList && userSimilarityList.length > 0)) {
            createUserSimilarityList();
        }
        // find a number of similar users
        List<Map.Entry<Integer, Double>> nns = new ArrayList<>();
        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];

        int count = 0;
        Set<Integer> userSet = trainMatrix.getRowsSet(itemIdx);
        for (Map.Entry<Integer, Double> userRatingEntry : simList) {
            int similarUserIdx = userRatingEntry.getKey();
            if (!userSet.contains(similarUserIdx)) {
                continue;
            }
            double sim = userRatingEntry.getValue();
            if (isRanking) {
                nns.add(userRatingEntry);
                count++;
            } else if (sim > 0) {
                nns.add(userRatingEntry);
                count++;
            }
            if (count == knn) {
                break;
            }
        }

        if (nns.size() == 0) {
            return isRanking ? 0 : globalMean;
        }

        if (isRanking) {
            double sum = 0.0d;
            for (Entry<Integer, Double> userRatingEntry : nns) {
                sum += userRatingEntry.getValue();
            }
            return sum;
        } else {
            // for rating prediction
            double sum = 0, ws = 0;
            for (Entry<Integer, Double> userRatingEntry : nns) {
                int similarUserIdx = userRatingEntry.getKey();
                double sim = userRatingEntry.getValue();
                double rate = trainMatrix.get(similarUserIdx, itemIdx);
                sum += sim * (rate - userMeans.get(similarUserIdx));
                ws += Math.abs(sim);
            }
            return ws > 0 ? userMeans.get(userIdx) + sum / ws : globalMean;
        }
    }

    public void createUserSimilarityList() {
        userSimilarityList = new ArrayList[numUsers];
        for (int userIndex = 0; userIndex < numUsers; ++userIndex) {
            SparseVector similarityVector = similarityMatrix.row(userIndex);
            userSimilarityList[userIndex] = new ArrayList<>(similarityVector.size());
            Iterator<VectorEntry> simItr = similarityVector.iterator();
            while (simItr.hasNext()) {
                VectorEntry simVectorEntry = simItr.next();
                userSimilarityList[userIndex].add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
            }
            Lists.sortList(userSimilarityList[userIndex], true);
        }
    }
}

