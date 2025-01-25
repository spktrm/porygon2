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
goog.exportSymbol('proto.FeatureAbsoluteEdge', null, global);
goog.exportSymbol('proto.FeatureEntity', null, global);
goog.exportSymbol('proto.FeatureMoveset', null, global);
goog.exportSymbol('proto.FeatureRelativeEdge', null, global);
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
  MOVESET_SPECIES_ID: 7,
  MOVESET_ENTITY_INDEX: 8
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
proto.FeatureAbsoluteEdge = {
  EDGE_TURN_ORDER_VALUE: 0,
  EDGE_TYPE_TOKEN: 1,
  EDGE_WEATHER_ID: 2,
  EDGE_WEATHER_MIN_DURATION: 3,
  EDGE_WEATHER_MAX_DURATION: 4,
  EDGE_TERRAIN_ID: 5,
  EDGE_TERRAIN_MIN_DURATION: 6,
  EDGE_TERRAIN_MAX_DURATION: 7,
  EDGE_PSEUDOWEATHER_ID: 8,
  EDGE_PSEUDOWEATHER_MIN_DURATION: 9,
  EDGE_PSEUDOWEATHER_MAX_DURATION: 10,
  EDGE_REQUEST_COUNT: 11,
  EDGE_VALID: 12,
  EDGE_INDEX: 13,
  EDGE_TURN_VALUE: 14
};

/**
 * @enum {number}
 */
proto.FeatureRelativeEdge = {
  EDGE_MAJOR_ARG: 0,
  EDGE_MINOR_ARG0: 1,
  EDGE_MINOR_ARG1: 2,
  EDGE_MINOR_ARG2: 3,
  EDGE_MINOR_ARG3: 4,
  EDGE_ACTION_TOKEN: 5,
  EDGE_ITEM_TOKEN: 6,
  EDGE_ABILITY_TOKEN: 7,
  EDGE_NUM_FROM_TYPES: 8,
  EDGE_FROM_TYPE_TOKEN0: 9,
  EDGE_FROM_TYPE_TOKEN1: 10,
  EDGE_FROM_TYPE_TOKEN2: 11,
  EDGE_FROM_TYPE_TOKEN3: 12,
  EDGE_FROM_TYPE_TOKEN4: 13,
  EDGE_NUM_FROM_SOURCES: 14,
  EDGE_FROM_SOURCE_TOKEN0: 15,
  EDGE_FROM_SOURCE_TOKEN1: 16,
  EDGE_FROM_SOURCE_TOKEN2: 17,
  EDGE_FROM_SOURCE_TOKEN3: 18,
  EDGE_FROM_SOURCE_TOKEN4: 19,
  EDGE_DAMAGE_RATIO: 20,
  EDGE_HEAL_RATIO: 21,
  EDGE_EFFECT_TOKEN: 22,
  EDGE_BOOST_ATK_VALUE: 23,
  EDGE_BOOST_DEF_VALUE: 24,
  EDGE_BOOST_SPA_VALUE: 25,
  EDGE_BOOST_SPD_VALUE: 26,
  EDGE_BOOST_SPE_VALUE: 27,
  EDGE_BOOST_ACCURACY_VALUE: 28,
  EDGE_BOOST_EVASION_VALUE: 29,
  EDGE_STATUS_TOKEN: 30,
  EDGE_SIDECONDITIONS0: 31,
  EDGE_SIDECONDITIONS1: 32,
  EDGE_TOXIC_SPIKES: 33,
  EDGE_SPIKES: 34
};

goog.object.extend(exports, proto);
