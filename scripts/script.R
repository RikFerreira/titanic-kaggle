library(tidyverse)
library(naniar)
library(scales)
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)

raw <- read_csv("./data/train.csv") %>%
  mutate(Survived = ifelse(Survived == 1, TRUE, FALSE)) %>%
  mutate(Survived = as.factor(Survived))

head(raw)

glimpse(raw)

vis_miss(raw)

raw %>%
  count(Pclass, Survived) %>%
  mutate(
    nrel = n / sum(n)
  ) %>%
  ggplot() +
  geom_col(aes(x = Pclass, y = nrel, fill = Survived)) +
  scale_y_continuous(labels = percent)

raw %>%
  count(Sex, Survived) %>%
  mutate(
    nrel = n / sum(n)
  ) %>%
  ggplot() +
  geom_col(aes(x = Sex, y = nrel, fill = Survived)) +
  scale_y_continuous(labels = percent)

raw %>%
  ggplot() +
  geom_boxplot(aes(x = Survived, y = Age, group = Survived))

raw %>%
  ggplot() +
  geom_histogram(aes(x = Age, fill = Survived))

raw %>%
  mutate(hasSib = ifelse(SibSp > 0, TRUE, FALSE)) %>%
  count(hasSib, Survived) %>%
  ggplot() +
  geom_col(aes(x = hasSib, y = n, fill = Survived))

raw %>%
  mutate(hasPar = ifelse(Parch > 0, TRUE, FALSE)) %>%
  count(hasPar, Survived) %>%
  ggplot() +
  geom_col(aes(x = hasPar, y = n, fill = Survived))

raw %>%
  group_by(SibSp, Survived) %>%
  summarise(meanAge = mean(Age, na.rm = TRUE)) %>%
  pivot_wider(
    names_from = Survived,
    values_from = meanAge
  )

raw %>%
  ggplot() +
  geom_boxplot(aes(x = Survived, y = Fare, group = Survived))

t.test(Fare ~ Survived, data = raw %>% mutate(Fare = scale(Fare)), na.action = na.omit)

raw %>%
  count(Embarked, Survived) %>%
  mutate(
    nrel = n / sum(n)
  ) %>%
  ggplot() +
  geom_col(aes(x = Embarked, y = nrel, fill = Survived)) +
  scale_y_continuous(labels = percent)

train_treated <- raw %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age))

# dec_tree <- rpart(Survived ~ Pclass + Sex + SibSp + Parch, data = train_treated, method = "class")
dec_tree <- rpart(Survived ~ ., data = train_treated, method = "class")

rpart.plot(dec_tree)

test <- read_csv("./data/test.csv")

test_treated <- test %>%
  select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age))

pred <- predict(dec_tree, train_treated, type = "class")

confusionMatrix(pred, train_treated$Survived)

train_predicted <- bind_cols(.pred = pred, train_treated)

pred_roc <- prediction(as.numeric(pred), as.numeric(train_treated$Survived))
performance <- performance(pred_roc, "tpr", "fpr", measure = "auc")
plot(performance)
performance@y.values

test_pred <- bind_cols(
  PassengerId = test$PassengerId,
  Survived = predict(dec_tree, test_treated, type = "class")
) %>%
  mutate(
    Survived = ifelse(Survived == "FALSE", 0, 1)
  )

write_csv(test_pred, "./output/submission.csv")