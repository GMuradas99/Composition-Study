import {trials} from './trials.js';

function extractKeyValues(data, targetKey) {
  const result = {};

  for (const [imageName, properties] of Object.entries(data)) {
    // Pick the target key value, defaulting to 0 or undefined if it doesn't exist
    result[imageName] = properties[targetKey] ?? 0;
  }

  return result;
}

function sortAndRandomizeTies(extractedData) {
  // 1. Convert object to [image, count] pairs
  const entries = Object.entries(extractedData);

  // 2. Fisher-Yates Shuffle (randomize order of all pairs first)
  for (let i = entries.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [entries[i], entries[j]] = [entries[j], entries[i]];
  }

  // 3. Sort by count (e.g., ascending order; change to (b[1] - a[1]) for descending)
  entries.sort((a, b) => a[1] - b[1]);

  // 4. Reconstruct and return as an Object
  return Object.fromEntries(entries);
}

function choose_trial() {
    const IMAGES_TO_CHOOSE = 20;

    const THRESHOLD_ART = 50;
    const THRESHOLD_NAT = 50;

    const THRESHOLD_TRIAL_TYPE = 10;

    const ret = {}

    // Choosing the image type based on the number of participants in each trial
    let image_type = null;
    if (trials.Art.Num_Participants >= THRESHOLD_ART && trials.Nat.Num_Participants < THRESHOLD_NAT) {
        image_type = 'Nat';
    }
    else if (trials.Nat.Num_Participants >= THRESHOLD_NAT && trials.Art.Num_Participants < THRESHOLD_ART) {
        image_type = 'Art';
    }
    else{
        // Randomly choose between 'Art' and 'Natural' if both thresholds are met
        image_type = Math.random() < 0.5 ? 'Art' : 'Nat';
    }
    ret.image_type = image_type;

    // Choosing the trial based on the number of participants in each trial
    let trial_type = null;
    const trial_types = trials[image_type].Trial_Participants;
    const keysBelowThreshold = Object.entries(trial_types).filter(([key, value]) => value < THRESHOLD_TRIAL_TYPE).map(([key, value]) => key);
    ret.trial_type = keysBelowThreshold[Math.floor(Math.random() * keysBelowThreshold.length)];

    // Choosing the images
    const image_counts = trials[image_type].Image_Participants;
    const images_trial_counts = extractKeyValues(image_counts, ret.trial_type);
    const sorted_images = sortAndRandomizeTies(images_trial_counts);
    const selected_images = Object.keys(sorted_images).slice(0, IMAGES_TO_CHOOSE);

    ret.selected_images = selected_images;
    
    return ret;
}

const choice = choose_trial();
console.log(choice);

// console.log(choose_trial());
