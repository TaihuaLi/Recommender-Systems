package net.librec.recommender.hybrid;

import java.beans.FeatureDescriptor;
import java.io.*;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import com.google.common.collect.*;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.AbstractRecommender;

public class NaiveBayesRecommender extends AbstractRecommender
{
    private static final int BSIZE = 1024 * 1024;

    protected SparseMatrix m_featureMatrix;
    protected double m_threshold;
    protected DenseMatrix PLF; // p(feature|like)
    protected DenseMatrix PDLF; // p(feature|~like)
    protected double [] PL; // p(like)
    protected double [] PDL; // p(~like)
    private BiMap<String, Integer> itemIds, featureIds; // user/feature {raw id, inner id} map

	/**
	 * Set up code expects a file name under the parameter dfs.content.path. It loads
	 * comma-separated feature data and stores a row at a time in the contentTable.
	 * After the data is loaded, it is converted into a SparseMatrix and stored in
	 * m_featureMatrix. This code borrows heavily from the implementation in
	 * net.librec.data.convertor.TextDataConvertor.
	**/
    @Override
    public void setup() throws LibrecException {
        super.setup();

        String contentPath = conf.get("dfs.content.path");
        Table<Integer, Integer, Integer> contentTable = HashBasedTable.create();
        HashBiMap<String, Integer> itemIds = HashBiMap.create();
        HashBiMap<String, Integer> featureIds = HashBiMap.create();

        int rowCount = 0;
        int maxFeature = -1;

        try {
            FileInputStream fileInputStream = new FileInputStream(contentPath);
            FileChannel fileRead = fileInputStream.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
            int len;
            String bufferLine = new String();
            byte[] bytes = new byte[BSIZE];
            while ((len = fileRead.read(buffer)) != -1) {
                buffer.flip();
                buffer.get(bytes, 0, len);
                bufferLine = bufferLine.concat(new String(bytes, 0, len));
                // String spl = System.getProperty("line.separator");
                String[] bufferData = bufferLine.split(System.getProperty("line.separator") + "+");
                boolean isComplete = bufferLine.endsWith(System.getProperty("line.separator"));
                int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
                for (int i = 0; i < loopLength; i++) {
                    String line = new String(bufferData[i]);
                    String[] data = line.trim().split("[ \t,]+");

                    String item = data[0];
                    // inner id starting from 0
                    int row = itemIds.containsKey(item) ? itemIds.get(item) : itemIds.size();
                    itemIds.put(item, row);

                    for (int j = 1; j < data.length; j++) {
                        String feature = data[j];
                        int col = featureIds.containsKey(feature) ? featureIds.get(feature) : featureIds.size();
                        featureIds.put(feature, col);

                        contentTable.put(row, col, 1);
                    }

                }
            }

            // System.out.println(trainMatrix.row(1));

        } catch (IOException e) {
            LOG.error("Error reading file: " + contentPath + e);
            throw (new LibrecException(e));
        }

        m_featureMatrix = new SparseMatrix(itemIds.size(), featureIds.size(), contentTable);
        LOG.info("Loaded item features from " + contentPath);

        // System.out.println(m_featureMatrix.row(1));
    }

    @Override
    public void trainModel() throws LibrecException {

        // System.out.println(m_featureMatrix);
        // m_featureMatrix  user  feature
        m_threshold = conf.getDouble("rec.rating.threshold");
        int NumUser = trainMatrix.numRows(); // each row represents a user;  user, item, rating
        int NumFeat = m_featureMatrix.numColumns(); // each column represents a feature

        PLF = new DenseMatrix(NumUser, NumFeat); // p(feature|like)
        PDLF = new DenseMatrix(NumUser, NumFeat); // p(feature|~like)
        PL = new double[NumUser]; // p(like)
        PDL = new double[NumUser]; // p(~like)

        int[] FeatPerLike = new int[NumFeat];      // placeholder for number of feature per like (for P(xi|c(X) = 1)
        int[] FeatPerDislike = new int[NumFeat];   // placeholder for number of feature per dislike (for P(xi|c(X) = -1)
        int like;                              // counter for number of liked
        int dislike;                           // counter for number of disliked

        // for each user
        for (int user = 0; user < NumUser; user++) {

            // reset placeholder and counter
            for (int f = 0; f < NumFeat; f++) {
                FeatPerLike[f] = 0;
                FeatPerDislike[f] = 0;
            }
            like = 0;
            dislike = 0;

            // for each item
            SparseVector items = trainMatrix.row(user);      // get rated items of the user
            // System.out.println(items);
            List<Integer> indexes = items.getIndexList();   // list of indexes for items
            for (int ind: indexes) {

                double rating = items.get(ind);             // get rating of an item
                // System.out.println(rating);
                if (rating > m_threshold){                  // get the number of (dis)liked
                    like++;
                } else if (rating > 0) {
                    dislike++;
                }

                // for each feature with the corresponding item.get(ind)
                SparseVector features = m_featureMatrix.row(ind);  // for each item, get features
                // System.out.println(features);
                List<Integer> FeatIndexes = features.getIndexList();  // list of indexes for features
                for (int f: FeatIndexes) {
                    if (rating > m_threshold){                     // get the number of (dis)like per feature
                        FeatPerLike[f]++;
                    } else if (rating > 0) {
                        FeatPerDislike[f]++;
                    }
                }
            }

            // like is an int : number of liked movies in the dataset by this user
            // dislike is an int : number of disliked movies in the dataset by this user
            // likePerFeat is an int array, index by feature: frequency of a feature appeared when this user likes
            // dislikePerFeat is an int array, index by feature :  frequency of a feature appeared when this user dislikes

            // laplace smoothing with alpha = 1 [look up 4.11 equation on page 155]

            // calculate ln(P(like) / P(~like))
            like = like + 1;
            dislike = dislike + 1;
            int total = like + dislike;

            PL[user] = (double) like/total; // p(like)
            PDL[user] = (double) dislike/total; // p(like)

            // calculate p(feature|like)  &   p(feature|dislike)
            for (int f = 0; f < NumFeat; f++) {
                double pfl = (FeatPerLike[f] + 1.0) / like;  // p(feature|like)
                double pfdl = (FeatPerDislike[f] + 1.0) / dislike; // p(feature|~like)

                PLF.set(user, f, pfl);
                PDLF.set(user, f, pfdl);
            }
        }
    }

    @Override
    public double predict(int user, int item) throws LibrecException {

        SparseVector features = m_featureMatrix.row(item);
        // System.out.println(features);
        List<Integer> FeatIndexes = features.getIndexList();

        double PPFL = PL[user];  // P(like)
        double PPFDL = PDL[user]; // P(~like)

        for (int f: FeatIndexes) {
            PPFL *= PLF.get(user, f); // P(like) * P(feature|like)
            PPFDL *= PDLF.get(user, f); // P(~like) * P(feature|~like)
        }

        // calcualte constant K
        double K = 1/(PPFL+ PPFDL); // k = 1 / ( P(feature|like)P(like) + P(feature|~like)P(~like))

        double Lprob = K * PPFL; // p(like|feature) = K * P(like) * P(feature|like)
        double DLprob = K * PPFDL; // p(~like|feature) = K * P(~like) * P(feature|~like)
        double logit = Math.log(Lprob / DLprob); // if logit > 0, then classified as liked

        // convert to prob
        double finalprob = Math.exp(logit) / (1 + Math.exp(logit));

        return minRate + finalprob * (maxRate - minRate); // predicted rating
    }
}