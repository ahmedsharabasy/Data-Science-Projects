


alter view view_Dim_shipMethod
as
SELECT TOP (1000) [ShipMethodID]
	  ,[Name]
      ,[ShipBase]
  FROM [AdventureWorks2022].[Purchasing].[ShipMethod]

