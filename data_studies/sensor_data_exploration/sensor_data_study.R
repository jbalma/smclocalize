library(tidyverse)
library(lubridate)
library(jsonlite)
library(igraph)

#Read in sensor data

sensor_data <- fromJSON('./input_data/sensor_data.json') %>% as.data.frame()

#Create unique sensor id's

sensor_data <- sensor_data %>%
  unite("loc_id", c("local_type", "local_id"), remove = FALSE) %>%
  unite("rem_id", c("remote_type", "remote_id"), remove = FALSE)

#Let's see if we can learn anything about the average distance to the various other sensors. We'll define a function that takes in the RSSI signal and gives us back a distance estimate.

rssi_to_dist <- function(rssi){
  intercept = -64.0
  slope = -20.0
  return(
    10 ^ ((rssi - intercept) / slope)
  )
}

sensor_data <- sensor_data %>%
  mutate(dist = rssi_to_dist(rssi),
         time = ymd_hms(observed_at))

#Let's pick a random child, and look at all his/her interactions. I'm going to pick child 11060, just because he/she appears in the initial sensor readings.

plot_child_ts <- function(sensor_data, loc_id, save_flag = FALSE){
  selected_child = loc_id
  
  distinct_loc_ids <- sensor_data %>%
    distinct(loc_id)
  
  distinct_times <- sensor_data %>%
    distinct(time)
  
  distinct_id_time <- expand.grid(loc_id = distinct_loc_ids$loc_id,
                                  rem_id = distinct_loc_ids$loc_id,
                                  time = distinct_times$time,
                                  stringsAsFactors = FALSE)
  
  new_sensor_data <- left_join(distinct_id_time, sensor_data) %>%
    arrange(remote_type)
  
  p <- ggplot(filter(new_sensor_data, loc_id == selected_child), aes(x = time, y = dist)) +
    geom_line(color = 'blue') +
    facet_wrap(~rem_id)
  if(save_flag){
    ggsave(filename = paste('./graphics/', selected_child, '_ts'), device = "pdf")
  }
  return(p)
}

plot_child_ts(sensor_data, "child_11060")

#Pretty noisy. It seems like there's probably a lot more information about proximity from the frequency of pings than there is from the strength of the rssi signal.

#Let's try looking at it as a distribution instead.

ggplot(filter(sensor_data, loc_id == "child_11060"), aes(x = dist)) +
  geom_histogram() +
  facet_wrap(~rem_id)

#This looks a bit more informative, since now we have ping counts and distance information together. Are we seeing particular children with which child_11060 played? Particular materials, and an area near which the child interacted with those materials?

#Let's try a social graph of the children and teachers. The nodes of the graph will be the children themselves. The edges we will weight according to the number of sensor pings between the children, the pings themselves will be weighted according to the estimated distance^-2 (an arbitrary choice).

children <- sensor_data %>% 
  filter(local_type %in% c("child", "teacher"), remote_type == c("child", "teacher")) %>%
  distinct(loc_id) %>%
  rename(child = loc_id)

interactions <- sensor_data %>%
  filter(local_type %in% c("child", "teacher"), remote_type == c("child", "teacher")) %>%
  group_by(loc_id, rem_id) %>%
  summarise(pings = n(),
            ping_weight = 1 / (mean(dist))^2) %>%
  mutate(weight = pings * ping_weight) %>%
  rename(to = loc_id, from = rem_id)

school_graph <- graph_from_data_frame(interactions, directed = FALSE, vertices = children)

sizes <- strength(school_graph) / 3

plot(school_graph,
     vertex.size = sizes,
     layout = layout_with_fr(school_graph, weights = interactions$weight),
     margin = c(-.75, -.75, -.75, -.75),
     vertex.label.cex = .5)

#There are some interesting differences here. child_47423 was either off on his/her own for most of the class, or had a malfunctioning sensor. A similar story for teacher_31986.

#There does seem to be a significant range of pings among the children. Let's look at the distribution to see how it looks.

ping_dist <- sensor_data %>%
  group_by(loc_id, local_type) %>%
  summarise(pings = n()) %>%
  arrange(pings)

ggplot(ping_dist, aes(x = pings, fill = as.factor(local_type))) +
  geom_histogram() +
  facet_wrap(~local_type)

#There were two children with dramatically fewer pings than their peers, child_11062 & child 47423. Perhaps they came late or left early? Let's take a look at the time series of their pings.

plot_child_ts(sensor_data, "child_11062")

#child_11062 does appear either ot have left early, or perhaps the sensor got turned off about a third of the way through the period.

plot_child_ts(sensor_data, "child_47423")

#A similar story for child_47423. Late arrival and early departure, or a sensor that cut out for significant chunks of the period.

#Now let's contrast that with child_11061, who actually had the most sensor pings of the children.
plot_child_ts(sensor_data, "child_11061")

#Is child_11061 friends with child_62459? That's what the consistent, low-distance pings would seem to suggest. The child also seems to have spent a fair amount of time with teacher_5612, and in area_11. Let's take a look at child_62459 to see if the relationship is recorded reciprocally.
plot_child_ts(sensor_data, "child_62459")

#There does indeed seem to be a reciprocal relationship recorded from child_62459's perspective. They apparently played together near area_11, apparently. Interesting that we don't see as much interaction with teacher_5612 from child_62459's perspective, though. Did they also both also have some contact early in the period with child_11060?