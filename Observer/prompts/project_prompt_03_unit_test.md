# Please write me one unit test, so I can understand and debug how system works.
* The interested flow is: list[StateVector] -> IFeatureEncoder -> list[FeatureVector] -> IAnomalyDetector.fit
* So I can  debug how FeatureVector is created, and how IAnomalyDetector use it with fit function.
* Keep it simple.
* list[StateVector] - should contain 3 vectors for 3 different example objects with same t.