// source: features.proto
/**
 * @fileoverview
 * @enhanceable
 * @suppress {missingRequire} reports error on implicit type usages.
 * @suppress {messageConventions} JS Compiler reports an error if a variable or
 *     field starts with 'MSG_' and isn't a translatable message.
 * @public
 */
// GENERATED CODE -- DO NOT EDIT!
/* eslint-disable */
// @ts-nocheck

var jspb = require('google-protobuf');
var goog = jspb;
var global = (function() { return this || window || global || self || Function('return this')(); }).call(null);

goog.exportSymbol('proto.FeatureEntity', null, global);
goog.exportSymbol('proto.FeatureMoveset', null, global);
goog.exportSymbol('proto.FeatureTurnContext', null, global);
goog.exportSymbol('proto.FeatureWeather', null, global);
/**
 * @enum {number}
 */
proto.FeatureEntity = {
  SPECIES: 0,
  ITEM: 1,
  ITEM_EFFECT: 2,
  ABILITY: 3,
  GENDER: 4,
  ACTIVE: 5,
  FAINTED: 6,
  HP: 7,
  MAXHP: 8,
  STATUS: 9,
  TOXIC_TURNS: 10,
  SLEEP_TURNS: 11,
  BEING_CALLED_BACK: 12,
  TRAPPED: 13,
  NEWLY_SWITCHED: 14,
  LEVEL: 15,
  MOVEID0: 16,
  MOVEID1: 17,
  MOVEID2: 18,
  MOVEID3: 19,
  MOVEPP0: 20,
  MOVEPP1: 21,
  MOVEPP2: 22,
  MOVEPP3: 23
};

/**
 * @enum {number}
 */
proto.FeatureMoveset = {
  MOVEID: 0,
  PPLEFT: 1,
  PPMAX: 2
};

/**
 * @enum {number}
 */
proto.FeatureTurnContext = {
  VALID: 0,
  IS_MY_TURN: 1,
  ACTION: 2,
  MOVE: 3,
  SWITCH_COUNTER: 4,
  MOVE_COUNTER: 5,
  TURN: 6
};

/**
 * @enum {number}
 */
proto.FeatureWeather = {
  WEATHER_ID: 0,
  MIN_DURATION: 1,
  MAX_DURATION: 2
};

goog.object.extend(exports, proto);
