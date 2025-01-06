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

goog.exportSymbol('proto.EdgeFromTypes', null, global);
goog.exportSymbol('proto.EdgeTypes', null, global);
goog.exportSymbol('proto.FeatureEdge', null, global);
goog.exportSymbol('proto.FeatureEntity', null, global);
goog.exportSymbol('proto.FeatureMoveset', null, global);
goog.exportSymbol('proto.FeatureTurnContext', null, global);
goog.exportSymbol('proto.FeatureWeather', null, global);
goog.exportSymbol('proto.MovesetActionType', null, global);
/**
 * @enum {number}
 */
proto.FeatureEntity = {
  ENTITY_SPECIES: 0,
  ENTITY_ITEM: 1,
  ENTITY_ITEM_EFFECT: 2,
  ENTITY_ABILITY: 3,
  ENTITY_GENDER: 4,
  ENTITY_ACTIVE: 5,
  ENTITY_FAINTED: 6,
  ENTITY_HP: 7,
  ENTITY_MAXHP: 8,
  ENTITY_HP_RATIO: 9,
  ENTITY_STATUS: 10,
  ENTITY_TOXIC_TURNS: 11,
  ENTITY_SLEEP_TURNS: 12,
  ENTITY_BEING_CALLED_BACK: 13,
  ENTITY_TRAPPED: 14,
  ENTITY_NEWLY_SWITCHED: 15,
  ENTITY_LEVEL: 16,
  ENTITY_MOVEID0: 17,
  ENTITY_MOVEID1: 18,
  ENTITY_MOVEID2: 19,
  ENTITY_MOVEID3: 20,
  ENTITY_MOVEPP0: 21,
  ENTITY_MOVEPP1: 22,
  ENTITY_MOVEPP2: 23,
  ENTITY_MOVEPP3: 24,
  ENTITY_HAS_STATUS: 25,
  ENTITY_BOOST_ATK_VALUE: 26,
  ENTITY_BOOST_DEF_VALUE: 27,
  ENTITY_BOOST_SPA_VALUE: 28,
  ENTITY_BOOST_SPD_VALUE: 29,
  ENTITY_BOOST_SPE_VALUE: 30,
  ENTITY_BOOST_ACCURACY_VALUE: 31,
  ENTITY_BOOST_EVASION_VALUE: 32,
  ENTITY_VOLATILES0: 33,
  ENTITY_VOLATILES1: 34,
  ENTITY_VOLATILES2: 35,
  ENTITY_VOLATILES3: 36,
  ENTITY_VOLATILES4: 37,
  ENTITY_VOLATILES5: 38,
  ENTITY_VOLATILES6: 39,
  ENTITY_VOLATILES7: 40,
  ENTITY_VOLATILES8: 41,
  ENTITY_SIDE: 42,
  ENTITY_TYPECHANGE0: 43,
  ENTITY_TYPECHANGE1: 44
};

/**
 * @enum {number}
 */
proto.MovesetActionType = {
  MOVESET_ACTION_TYPE_MOVE: 0,
  MOVESET_ACTION_TYPE_SWITCH: 1
};

/**
 * @enum {number}
 */
proto.FeatureMoveset = {
  MOVESET_ACTION_ID: 0,
  MOVESET_PPUSED: 1,
  MOVESET_LEGAL: 2,
  MOVESET_SIDE: 3,
  MOVESET_ACTION_TYPE: 4,
  MOVESET_EST_DAMAGE: 5,
  MOVESET_MOVE_ID: 6,
  MOVESET_SPECIES_ID: 7
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

/**
 * @enum {number}
 */
proto.EdgeTypes = {
  EDGE_TYPE_NONE: 0,
  MOVE_EDGE: 1,
  SWITCH_EDGE: 2,
  EFFECT_EDGE: 3,
  CANT_EDGE: 4,
  EDGE_TYPE_START: 5
};

/**
 * @enum {number}
 */
proto.EdgeFromTypes = {
  EDGE_FROM_NONE: 0,
  EDGE_FROM_ITEM: 1,
  EDGE_FROM_EFFECT: 2,
  EDGE_FROM_MOVE: 3,
  EDGE_FROM_ABILITY: 4,
  EDGE_FROM_SIDECONDITION: 5,
  EDGE_FROM_STATUS: 6,
  EDGE_FROM_WEATHER: 7,
  EDGE_FROM_TERRAIN: 8,
  EDGE_FROM_PSEUDOWEATHER: 9
};

/**
 * @enum {number}
 */
proto.FeatureEdge = {
  TURN_ORDER_VALUE: 0,
  EDGE_TYPE_TOKEN: 1,
  MAJOR_ARG: 2,
  MINOR_ARG: 3,
  ACTION_TOKEN: 4,
  ITEM_TOKEN: 5,
  ABILITY_TOKEN: 6,
  FROM_TYPE_TOKEN: 7,
  FROM_SOURCE_TOKEN: 8,
  DAMAGE_RATIO: 9,
  HEAL_RATIO: 10,
  EFFECT_TOKEN: 11,
  BOOST_ATK_VALUE: 12,
  BOOST_DEF_VALUE: 13,
  BOOST_SPA_VALUE: 14,
  BOOST_SPD_VALUE: 15,
  BOOST_SPE_VALUE: 16,
  BOOST_ACCURACY_VALUE: 17,
  BOOST_EVASION_VALUE: 18,
  STATUS_TOKEN: 19,
  EDGE_AFFECTING_SIDE: 20,
  PLAYER_ID: 21,
  REQUEST_COUNT: 22,
  EDGE_VALID: 23,
  EDGE_INDEX: 24,
  TURN_VALUE: 25
};

goog.object.extend(exports, proto);
