package net.librec.recommender;

import net.librec.common.LibrecException;
import net.librec.recommender.cf.rating.BiasedMFRecommender;
import net.librec.recommender.hybrid.NaiveBayesRecommender;

public class HybridRecommender extends AbstractRecommender{

    NaiveBayesRecommender NB = new NaiveBayesRecommender();
    BiasedMFRecommender BMF = new BiasedMFRecommender();
    double nb_w;
    double bmf_w;

    @Override
    public void setup() throws LibrecException {
        super.setup();
        context = getContext();

        // get wegiht paramters
        nb_w = conf.getDouble("rec.content.weight");
        bmf_w = 1 - nb_w;
        // System.out.println(bmf_w);

        // set up Biased MF and Naive Bayes model
        NB.setContext(context);
        NB.setup();

        BMF.setContext(context);
        BMF.setup();
    }

    @Override
    protected void trainModel() throws LibrecException {
        // train NB and GB models
        NB.trainModel();
        BMF.trainModel();
    }

    @Override
    protected double predict(int user, int item) throws LibrecException {
        // get weighted predictions
        double nb_prediction = NB.predict(user, item);
        double bmf_prediction = BMF.predict(user, item);
        return nb_prediction * nb_w + bmf_prediction * bmf_w;
    }
}
