


create view view_Dim_salesTerritory
as
SELECT TOP (1000) [TerritoryID]
      ,[Name]
      ,[CountryRegionCode]
      ,[Group]
      ,[SalesLastYear]
  FROM [AdventureWorks2022].[Sales].[SalesTerritory]
