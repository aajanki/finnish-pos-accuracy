package RaudikkoPipeline;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import com.google.gson.Gson;
import fi.evident.raudikko.Morphology;
import fi.evident.raudikko.Analyzer;

public class App {
    public static void main(String[] args) {
        Gson gson = new Gson();

        Morphology morphology = Morphology.loadBundled();
        Analyzer analyzer = morphology.newAnalyzer();

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        reader.lines()
            .forEach(word -> System.out.println(gson.toJson(analyzer.analyze(word))));
    }
}
