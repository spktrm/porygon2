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
goog.exportSymbol('proto.FeatureAdditionalInformation', null, global);
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
  ENTITY_STATUS: 9,
  ENTITY_TOXIC_TURNS: 10,
  ENTITY_SLEEP_TURNS: 11,
  ENTITY_BEING_CALLED_BACK: 12,
  ENTITY_TRAPPED: 13,
  ENTITY_NEWLY_SWITCHED: 14,
  ENTITY_LEVEL: 15,
  ENTITY_MOVEID0: 16,
  ENTITY_MOVEID1: 17,
  ENTITY_MOVEID2: 18,
  ENTITY_MOVEID3: 19,
  ENTITY_MOVEPP0: 20,
  ENTITY_MOVEPP1: 21,
  ENTITY_MOVEPP2: 22,
  ENTITY_MOVEPP3: 23,
  ENTITY_HAS_STATUS: 24,
  ENTITY_BOOST_ATK_VALUE: 25,
  ENTITY_BOOST_DEF_VALUE: 26,
  ENTITY_BOOST_SPA_VALUE: 27,
  ENTITY_BOOST_SPD_VALUE: 28,
  ENTITY_BOOST_SPE_VALUE: 29,
  ENTITY_BOOST_ACCURACY_VALUE: 30,
  ENTITY_BOOST_EVASION_VALUE: 31,
  ENTITY_VOLATILES0: 32,
  ENTITY_VOLATILES1: 33,
  ENTITY_VOLATILES2: 34,
  ENTITY_VOLATILES3: 35,
  ENTITY_VOLATILES4: 36,
  ENTITY_VOLATILES5: 37,
  ENTITY_VOLATILES6: 38,
  ENTITY_VOLATILES7: 39,
  ENTITY_VOLATILES8: 40,
  ENTITY_SIDE: 41,
  ENTITY_HP_TOKEN: 42
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
proto.FeatureAdditionalInformation = {
  NUM_FAINTED: 0,
  HP_TOTAL: 1,
  NUM_TYPES_PAD: 2,
  NUM_TYPES_UNK: 3,
  NUM_TYPES_BUG: 4,
  NUM_TYPES_DARK: 5,
  NUM_TYPES_DRAGON: 6,
  NUM_TYPES_ELECTRIC: 7,
  NUM_TYPES_FAIRY: 8,
  NUM_TYPES_FIGHTING: 9,
  NUM_TYPES_FIRE: 10,
  NUM_TYPES_FLYING: 11,
  NUM_TYPES_GHOST: 12,
  NUM_TYPES_GRASS: 13,
  NUM_TYPES_GROUND: 14,
  NUM_TYPES_ICE: 15,
  NUM_TYPES_NORMAL: 16,
  NUM_TYPES_POISON: 17,
  NUM_TYPES_PSYCHIC: 18,
  NUM_TYPES_ROCK: 19,
  NUM_TYPES_STEEL: 20,
  NUM_TYPES_STELLAR: 21,
  NUM_TYPES_WATER: 22,
  TOTAL_POKEMON: 23,
  WISHING: 24,
  MEMBER0_HP: 25,
  MEMBER1_HP: 26,
  MEMBER2_HP: 27,
  MEMBER3_HP: 28,
  MEMBER4_HP: 29,
  MEMBER5_HP: 30
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
  DAMAGE_TOKEN: 9,
  EFFECT_TOKEN: 10,
  BOOST_ATK_VALUE: 11,
  BOOST_DEF_VALUE: 12,
  BOOST_SPA_VALUE: 13,
  BOOST_SPD_VALUE: 14,
  BOOST_SPE_VALUE: 15,
  BOOST_ACCURACY_VALUE: 16,
  BOOST_EVASION_VALUE: 17,
  STATUS_TOKEN: 18,
  EDGE_AFFECTING_SIDE: 19,
  PLAYER_ID: 20,
  REQUEST_COUNT: 21,
  EDGE_VALID: 22,
  EDGE_INDEX: 23,
  TURN_VALUE: 24
};

goog.object.extend(exports, proto);
