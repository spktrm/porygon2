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

goog.exportSymbol('proto.AbsoluteEdgeFeature', null, global);
goog.exportSymbol('proto.EntityFeature', null, global);
goog.exportSymbol('proto.MovesetActionTypeEnum', null, global);
goog.exportSymbol('proto.MovesetFeature', null, global);
goog.exportSymbol('proto.MovesetHasPPEnum', null, global);
goog.exportSymbol('proto.RelativeEdgeFeature', null, global);
/**
 * @enum {number}
 */
proto.EntityFeature = {
  ENTITY_FEATURE___UNSPECIFIED: 0,
  ENTITY_FEATURE__SPECIES: 1,
  ENTITY_FEATURE__ITEM: 2,
  ENTITY_FEATURE__ITEM_EFFECT: 3,
  ENTITY_FEATURE__ABILITY: 4,
  ENTITY_FEATURE__GENDER: 5,
  ENTITY_FEATURE__ACTIVE: 6,
  ENTITY_FEATURE__FAINTED: 7,
  ENTITY_FEATURE__HP: 8,
  ENTITY_FEATURE__MAXHP: 9,
  ENTITY_FEATURE__HP_RATIO: 10,
  ENTITY_FEATURE__STATUS: 11,
  ENTITY_FEATURE__TOXIC_TURNS: 12,
  ENTITY_FEATURE__SLEEP_TURNS: 13,
  ENTITY_FEATURE__BEING_CALLED_BACK: 14,
  ENTITY_FEATURE__TRAPPED: 15,
  ENTITY_FEATURE__NEWLY_SWITCHED: 16,
  ENTITY_FEATURE__LEVEL: 17,
  ENTITY_FEATURE__MOVEID0: 18,
  ENTITY_FEATURE__MOVEID1: 19,
  ENTITY_FEATURE__MOVEID2: 20,
  ENTITY_FEATURE__MOVEID3: 21,
  ENTITY_FEATURE__MOVEPP0: 22,
  ENTITY_FEATURE__MOVEPP1: 23,
  ENTITY_FEATURE__MOVEPP2: 24,
  ENTITY_FEATURE__MOVEPP3: 25,
  ENTITY_FEATURE__HAS_STATUS: 26,
  ENTITY_FEATURE__BOOST_ATK_VALUE: 27,
  ENTITY_FEATURE__BOOST_DEF_VALUE: 28,
  ENTITY_FEATURE__BOOST_SPA_VALUE: 29,
  ENTITY_FEATURE__BOOST_SPD_VALUE: 30,
  ENTITY_FEATURE__BOOST_SPE_VALUE: 31,
  ENTITY_FEATURE__BOOST_ACCURACY_VALUE: 32,
  ENTITY_FEATURE__BOOST_EVASION_VALUE: 33,
  ENTITY_FEATURE__VOLATILES0: 34,
  ENTITY_FEATURE__VOLATILES1: 35,
  ENTITY_FEATURE__VOLATILES2: 36,
  ENTITY_FEATURE__VOLATILES3: 37,
  ENTITY_FEATURE__VOLATILES4: 38,
  ENTITY_FEATURE__VOLATILES5: 39,
  ENTITY_FEATURE__VOLATILES6: 40,
  ENTITY_FEATURE__VOLATILES7: 41,
  ENTITY_FEATURE__VOLATILES8: 42,
  ENTITY_FEATURE__SIDE: 43,
  ENTITY_FEATURE__TYPECHANGE0: 44,
  ENTITY_FEATURE__TYPECHANGE1: 45,
  ENTITY_FEATURE__ACTION_ID0: 46,
  ENTITY_FEATURE__ACTION_ID1: 47,
  ENTITY_FEATURE__ACTION_ID2: 48,
  ENTITY_FEATURE__ACTION_ID3: 49,
  ENTITY_FEATURE__NUM_MOVES: 50,
  ENTITY_FEATURE__IS_PUBLIC: 51
};

/**
 * @enum {number}
 */
proto.MovesetActionTypeEnum = {
  MOVESET_ACTION_TYPE_ENUM___UNSPECIFIED: 0,
  MOVESET_ACTION_TYPE_ENUM__MOVE: 1,
  MOVESET_ACTION_TYPE_ENUM__SWITCH: 2
};

/**
 * @enum {number}
 */
proto.MovesetHasPPEnum = {
  MOVESET_HAS_PP_ENUM___UNSPECIFIED: 0,
  MOVESET_HAS_PP_ENUM__YES: 1,
  MOVESET_HAS_PP_ENUM__NO: 2
};

/**
 * @enum {number}
 */
proto.MovesetFeature = {
  MOVESET_FEATURE___UNSPECIFIED: 0,
  MOVESET_FEATURE__ACTION_ID: 1,
  MOVESET_FEATURE__PP_RATIO: 2,
  MOVESET_FEATURE__MOVE_ID: 3,
  MOVESET_FEATURE__SPECIES_ID: 4,
  MOVESET_FEATURE__PP: 5,
  MOVESET_FEATURE__MAXPP: 6,
  MOVESET_FEATURE__HAS_PP: 7,
  MOVESET_FEATURE__ACTION_TYPE: 8
};

/**
 * @enum {number}
 */
proto.RelativeEdgeFeature = {
  RELATIVE_EDGE_FEATURE___UNSPECIFIED: 0,
  RELATIVE_EDGE_FEATURE__MAJOR_ARG: 1,
  RELATIVE_EDGE_FEATURE__MINOR_ARG0: 2,
  RELATIVE_EDGE_FEATURE__MINOR_ARG1: 3,
  RELATIVE_EDGE_FEATURE__MINOR_ARG2: 4,
  RELATIVE_EDGE_FEATURE__MINOR_ARG3: 5,
  RELATIVE_EDGE_FEATURE__ACTION_TOKEN: 6,
  RELATIVE_EDGE_FEATURE__ITEM_TOKEN: 7,
  RELATIVE_EDGE_FEATURE__ABILITY_TOKEN: 8,
  RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES: 9,
  RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0: 10,
  RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN1: 11,
  RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN2: 12,
  RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN3: 13,
  RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN4: 14,
  RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES: 15,
  RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0: 16,
  RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN1: 17,
  RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN2: 18,
  RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN3: 19,
  RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN4: 20,
  RELATIVE_EDGE_FEATURE__DAMAGE_RATIO: 21,
  RELATIVE_EDGE_FEATURE__HEAL_RATIO: 22,
  RELATIVE_EDGE_FEATURE__EFFECT_TOKEN: 23,
  RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE: 24,
  RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE: 25,
  RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE: 26,
  RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE: 27,
  RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE: 28,
  RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE: 29,
  RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE: 30,
  RELATIVE_EDGE_FEATURE__STATUS_TOKEN: 31,
  RELATIVE_EDGE_FEATURE__SIDECONDITIONS0: 32,
  RELATIVE_EDGE_FEATURE__SIDECONDITIONS1: 33,
  RELATIVE_EDGE_FEATURE__TOXIC_SPIKES: 34,
  RELATIVE_EDGE_FEATURE__SPIKES: 35,
  RELATIVE_EDGE_FEATURE__MOVE_TOKEN: 36
};

/**
 * @enum {number}
 */
proto.AbsoluteEdgeFeature = {
  ABSOLUTE_EDGE_FEATURE___UNSPECIFIED: 0,
  ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE: 1,
  ABSOLUTE_EDGE_FEATURE__TYPE_TOKEN: 2,
  ABSOLUTE_EDGE_FEATURE__WEATHER_ID: 3,
  ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION: 4,
  ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION: 5,
  ABSOLUTE_EDGE_FEATURE__TERRAIN_ID: 6,
  ABSOLUTE_EDGE_FEATURE__TERRAIN_MIN_DURATION: 7,
  ABSOLUTE_EDGE_FEATURE__TERRAIN_MAX_DURATION: 8,
  ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_ID: 9,
  ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MIN_DURATION: 10,
  ABSOLUTE_EDGE_FEATURE__PSEUDOWEATHER_MAX_DURATION: 11,
  ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT: 12,
  ABSOLUTE_EDGE_FEATURE__VALID: 13,
  ABSOLUTE_EDGE_FEATURE__INDEX: 14,
  ABSOLUTE_EDGE_FEATURE__TURN_VALUE: 15
};

goog.object.extend(exports, proto);
